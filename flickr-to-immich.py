"""
**** FLICKR EMMIGRATION TOOL ****
by Rob Brown
https://github.com/brownphotographic/Flickr-Emmigration-Tool

Copyright (C) 2025 Robert Brown

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

For usage instructions please see the README file.

"""

import bdb
from typing import Dict, List, Set, Optional, Tuple, Union
import time
from datetime import datetime
from enum import Enum
import json
import shutil
import subprocess
import logging
from pathlib import Path
import mimetypes
import argparse
from tqdm import tqdm
import os
import io
import flickrapi
import xml.etree.ElementTree as ET
from importlib import metadata
from tomlkit.api import key
from PIL import Image
import concurrent.futures

class MediaType(Enum):
    """Supported media types"""
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"

# Configuration flags
INCLUDE_EXTENDED_DESCRIPTION = True  # Set to False to only include original description
WRITE_XMP_SIDECARS = True  # Set to False to skip writing XMP sidecar files

class JPEGVerifier:
    """Utility class to verify and repair JPEG integrity"""

    @staticmethod
    def is_jpeg_valid(file_path: str) -> Tuple[bool, str]:
        """
        Verify if a JPEG file is valid by checking for proper markers.

        Args:
            file_path: Path to the JPEG file

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            with open(file_path, 'rb') as f:
                # Check SOI marker (Start of Image)
                if f.read(2) != b'\xFF\xD8':
                    return False, "Missing JPEG SOI marker"

                # Read file in chunks to find EOI marker
                # Using smaller chunks to be memory efficient
                chunk_size = 4096
                last_byte = None
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return False, "Missing JPEG EOI marker"

                    # Check for EOI marker, including split across chunks
                    if last_byte == b'\xFF' and chunk.startswith(b'\xD9'):
                        return True, ""

                    # Look for EOI marker within chunk
                    if b'\xFF\xD9' in chunk:
                        return True, ""

                    # Save last byte for next iteration
                    last_byte = chunk[-1:]

                    # If we're at the end of file without finding EOI
                    if len(chunk) < chunk_size:
                        return False, "Missing JPEG EOI marker"

        except Exception as e:
            return False, f"Error reading JPEG: {str(e)}"

    @staticmethod
    def attempt_repair(file_path: str) -> bool:
        """
        Attempt to repair a corrupted JPEG file using multiple methods.

        Args:
            file_path: Path to the JPEG file to repair

        Returns:
            bool: True if repair was successful, False otherwise
        """
        try:
            # Create a backup of the original file
            backup_path = file_path + '.bak'
            shutil.copy2(file_path, backup_path)
            logging.info(f"Created backup at {backup_path}")

            # Try multiple repair methods
            repair_methods = [
                ('PIL repair', JPEGVerifier._repair_using_pil),
                ('ExifTool repair', JPEGVerifier._repair_using_exiftool),
                ('EOI marker repair', JPEGVerifier._repair_by_adding_eoi)
            ]

            for method_name, repair_method in repair_methods:
                try:
                    logging.info(f"Attempting {method_name}...")

                    # Restore from backup before each attempt
                    shutil.copy2(backup_path, file_path)

                    # Try repair method
                    if repair_method(file_path):
                        # Verify the repaired file
                        is_valid, error_msg = JPEGVerifier.is_jpeg_valid(file_path)
                        if is_valid:
                            logging.info(f"Successfully repaired file using {method_name}")
                            os.remove(backup_path)  # Remove backup if repair succeeded
                            return True
                        else:
                            logging.warning(f"{method_name} failed verification: {error_msg}")
                    else:
                        logging.warning(f"{method_name} failed")

                except Exception as e:
                    logging.warning(f"{method_name} failed with error: {str(e)}")
                    continue

            # If all repair attempts failed, restore from backup
            logging.error("All repair attempts failed, restoring from backup")
            shutil.copy2(backup_path, file_path)
            os.remove(backup_path)
            return False

        except Exception as e:
            logging.error(f"Error in repair process: {str(e)}")
            # Try to restore from backup if it exists
            if os.path.exists(backup_path):
                try:
                    logging.info("Restoring from backup after error")
                    shutil.copy2(backup_path, file_path)
                    os.remove(backup_path)
                except Exception as restore_error:
                    logging.error(f"Failed to restore from backup: {str(restore_error)}")
            return False

    @staticmethod
    def _repair_using_pil(file_path: str) -> bool:
        """
        Attempt to repair using PIL by reading and rewriting the image.

        Args:
            file_path: Path to the JPEG file

        Returns:
            bool: True if repair was successful, False otherwise
        """
        try:
            with Image.open(file_path) as img:
                # Create a temporary buffer
                temp_buffer = io.BytesIO()

                # Save with maximum quality to preserve image data
                img.save(temp_buffer, format='JPEG', quality=100,
                        optimize=False, progressive=False)

                # Write back to file
                with open(file_path, 'wb') as f:
                    f.write(temp_buffer.getvalue())

                return True
        except Exception as e:
            logging.debug(f"PIL repair failed: {str(e)}")
            return False

    @staticmethod
    def _repair_using_exiftool(file_path: str) -> bool:
        """
        Attempt to repair using exiftool by stripping metadata and rewriting.

        Args:
            file_path: Path to the JPEG file

        Returns:
            bool: True if repair was successful, False otherwise
        """
        try:
            repair_args = [
                'exiftool',
                '-all=',  # Remove all metadata
                '-overwrite_original',
                '-ignoreMinorErrors',
                str(file_path)
            ]
            result = subprocess.run(repair_args, capture_output=True, text=True)

            if result.returncode == 0:
                return True
            else:
                logging.debug(f"ExifTool repair failed: {result.stderr}")
                return False
        except Exception as e:
            logging.debug(f"ExifTool repair failed: {str(e)}")
            return False

    @staticmethod
    def _repair_by_adding_eoi(file_path: str) -> bool:
        """
        Attempt to repair by adding EOI marker at the end of the file.
        This is a last resort method.

        Args:
            file_path: Path to the JPEG file

        Returns:
            bool: True if repair was successful, False otherwise
        """
        try:
            # First check if file already ends with EOI marker
            with open(file_path, 'rb') as f:
                f.seek(-2, 2)  # Seek to last 2 bytes
                if f.read(2) == b'\xFF\xD9':
                    return True  # Already has EOI marker

            # If not, append the EOI marker
            with open(file_path, 'ab') as f:
                f.write(b'\xFF\xD9')
            return True

        except Exception as e:
            logging.debug(f"EOI marker repair failed: {str(e)}")
            return False

    @staticmethod
    def verify_and_repair(file_path: str) -> Tuple[bool, str]:
        """
        Convenience method to verify and repair if necessary.

        Args:
            file_path: Path to the JPEG file

        Returns:
            Tuple[bool, str]: (success, message)
        """
        # First verify
        is_valid, error_msg = JPEGVerifier.is_jpeg_valid(file_path)
        if is_valid:
            return True, "File is valid"

        # Attempt repair if invalid
        logging.warning(f"JPEG validation failed: {error_msg}")
        if JPEGVerifier.attempt_repair(file_path):
            return True, "File repaired successfully"
        else:
            return False, "Could not repair file"

class FlickrToImmich:
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.webp',
        # Videos
        '.mp4', '.mov', '.avi', '.mpg', '.mpeg', '.m4v', '.3gp', '.wmv',
        '.webm', '.mkv', '.flv'
    }

    def _clear_output_directory(self):
        """Clear all contents of the output directory"""
        try:
            if self.output_dir.exists():
                logging.info(f"Clearing output directory: {self.output_dir}")
                shutil.rmtree(self.output_dir)
                self.output_dir.mkdir(parents=True)
                logging.info("Output directory cleared successfully")
        except Exception as e:
            raise ValueError(f"Error clearing output directory: {str(e)}")

    def _find_unorganized_photos(self) -> Dict[str, Path]:
        """
        Find all photos in the photos directory that aren't in any album
        Returns a dict mapping photo IDs to their file paths
        """
        try:
            # Get all media files in photos directory
            all_photos = {}  # photo_id -> Path
            unidentified_photos = []  # Files where we couldn't extract an ID

            logging.info("Scanning photos directory for unorganized photos...")
            for file in self.photos_dir.iterdir():
                if not file.is_file() or not any(file.name.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    continue

                photo_id = self._extract_photo_id(file.name)

                if photo_id:
                    all_photos[photo_id] = file
                else:
                    unidentified_photos.append(file)

            # For files without IDs, generate sequential IDs
            for idx, file in enumerate(unidentified_photos):
                generated_id = f"unknown_{idx+1}"
                all_photos[generated_id] = file
                logging.info(f"Generated ID {generated_id} for file: {file.name}")

            # Find photos not in any album
            organized_photos = set(self.photo_to_albums.keys())
            unorganized_photos = {}

            for photo_id, file_path in all_photos.items():
                if photo_id not in organized_photos:
                    unorganized_photos[photo_id] = file_path

            if unorganized_photos:
                logging.info(f"Found {len(unorganized_photos)} photos not in any album")
                logging.info("Sample of unorganized photos:")
                for photo_id, file_path in list(unorganized_photos.items())[:5]:
                    logging.info(f"  - {photo_id}: {file_path.name}")
            else:
                logging.info("All photos are organized in albums")

            return unorganized_photos

        except Exception as e:
            logging.error(f"Error finding unorganized photos: {str(e)}")
            return {}

    def _build_photo_album_mapping(self):
        """Build mapping of photos to their albums"""
        try:
            self.photo_to_albums = {}  # Initialize the dictionary

            # First process all album photos
            for album in self.albums:
                if 'photos' not in album:
                    logging.warning(f"Album '{album.get('title', 'Unknown')}' has no photos key")
                    continue

                for photo_id in album['photos']:
                    if photo_id not in self.photo_to_albums:
                        self.photo_to_albums[photo_id] = []
                    self.photo_to_albums[photo_id].append(album['title'])

            # Find and add unorganized photos to 00_NoAlbum
            unorganized_photos = self._find_unorganized_photos()
            if unorganized_photos:
                # Add to photo_to_albums mapping
                for photo_id in unorganized_photos:
                    self.photo_to_albums[photo_id] = ['00_NoAlbum']

                # Add 00_NoAlbum to self.albums
                no_album = {
                    'title': '00_NoAlbum',
                    'description': 'Photos not organized in any Flickr album',
                    'photos': list(unorganized_photos.keys())
                }
                self.albums.append(no_album)

                # Store the file paths for unorganized photos for later use
                self.unorganized_photo_paths = unorganized_photos

            total_photos = len(self.photo_to_albums)
            organized_count = sum(1 for p in self.photo_to_albums.values() if '00_NoAlbum' not in p)
            unorganized_count = len(unorganized_photos)

            logging.info(f"Photo organization summary:")
            logging.info(f"- Total photos: {total_photos}")
            logging.info(f"- In albums: {organized_count}")
            logging.info(f"- Not in albums: {unorganized_count}")

        except Exception as e:
            logging.error(f"Error building photo-album mapping: {str(e)}")
            self.photo_to_albums = {}  # Initialize empty if there's an error
            raise

    def __init__(self,
                    metadata_dir: str,
                    photos_dir: str,
                    output_dir: str,
                    api_key: Optional[str] = None,
                    log_file: Optional[str] = None,
                    results_dir: Optional[str] = None,
                    include_extended_description: bool = INCLUDE_EXTENDED_DESCRIPTION,
                    write_xmp_sidecars: bool = WRITE_XMP_SIDECARS,
                    block_if_failure: bool = False,
                    resume: bool = False,
                    quiet: bool = False):  # Added quiet parameter

            self.block_if_failure = block_if_failure
            self.resume = resume

            # Track processing statistics
            self.stats = {
                'total_files': 0,
                'successful': {
                    'count': 0,
                    'details': []  # Will store (source_file, dest_file, status) tuples
                },
                'failed': {
                    'metadata': {
                        'count': 0,
                        'details': []  # Will store (file, error_msg, exported) tuples
                    },
                    'file_copy': {
                        'count': 0,
                        'details': []  # Will store (file, error_msg) tuples
                    }
                },
                'skipped': {
                    'count': 0,
                    'details': []  # Will store (file, reason) tuples
                }
            }

            """
            Initialize the converter with source and destination directories

            Args:
                metadata_dir: Directory containing Flickr export JSON files
                photos_dir: Directory containing the photos and videos
                output_dir: Directory where the album structure will be created
                api_key: Optional Flickr API key for additional metadata
                log_file: Optional path to log file
                include_extended_description: Whether to include extended metadata in description
                write_xmp_sidecars: Whether to write XMP sidecar files
                quiet: Whether to reduce console output (default: False)
            """
            self.include_extended_description = include_extended_description
            self.write_xmp_sidecars = write_xmp_sidecars

            # Setup logging with quiet parameter
            self._setup_logging(log_file, quiet)

            # Initialize directories
            self.metadata_dir = Path(metadata_dir)
            self.photos_dir = Path(photos_dir)
            self.output_dir = Path(output_dir)
            self.results_dir = Path(results_dir) if results_dir else self.output_dir

            # Clear output directory if not resuming
            if not self.resume:
                self._clear_output_directory()

            # Initialize data containers
            self.account_data = None
            self.user_mapping = {}
            self.photo_to_albums: Dict[str, List[str]] = {}

    #       # Validate directories
    #       self._validate_directories()

            # Initialize Flickr API if key is provided
            self.flickr = None
            self.user_info_cache = {}
            if api_key:
                try:
                    self.flickr = flickrapi.FlickrAPI(api_key, '', format='etree')
                    logging.getLogger('flickrapi').setLevel(logging.ERROR)
                    logging.info("Successfully initialized Flickr API")
                except Exception as e:
                    logging.warning(f"Failed to initialize Flickr API: {e}")
                    logging.warning("User info lookup will be disabled")

            try:
                # Debug: List ALL files in metadata directory
                logging.info("=== DIRECTORY CONTENTS ===")
                logging.info(f"Metadata directory: {self.metadata_dir}")
                for file in self.metadata_dir.iterdir():
                    logging.info(f"Found file: {file.name}")
                logging.info("=== END DIRECTORY CONTENTS ===")

                # Load account profile (single file)
                self._load_account_profile()

                # Load all data, handling both single and multi-part files
                self.contacts = self._load_multipart_json("contacts", "contacts")
                self.comments = self._load_multipart_json("comments", "comments")
                self.favorites = self._load_multipart_json("faves", "faves")
                self.followers = self._load_multipart_json("followers", "followers")
                self.galleries = self._load_multipart_json("galleries", "galleries")
                self.apps_comments = self._load_multipart_json("apps_comments", "comments")
                self.gallery_comments = self._load_multipart_json("galleries_comments", "comments")

                # Process user mappings from loaded contacts data
                self._process_user_mappings()

                # Load albums (could be single or multi-part)
                self.albums = self._load_multipart_json("albums", "albums")
                if not isinstance(self.albums, list):
                    logging.error("Albums data is not in expected list format")
                    raise ValueError("Invalid albums data format")

                # Build photo -> album mapping
                self._build_photo_album_mapping()

            except Exception as e:
                logging.error(f"Initialization failed: {str(e)}")
                raise

    def _validate_directories(self):
        """Validate input and output directories"""
        if not self.metadata_dir.exists():
            raise ValueError(f"Metadata directory does not exist: {self.metadata_dir}")

        if not self.photos_dir.exists():
            raise ValueError(f"Photos directory does not exist: {self.photos_dir}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self, log_file: Optional[str], quiet: bool = False):
        """
        Configure logging with both file and console output

        Args:
            log_file: Optional path to log file
            quiet: If True, only show critical errors in console
        """
        # Delete existing log file if it exists
        if log_file and Path(log_file).exists():
            try:
                Path(log_file).unlink()
            except Exception as e:
                print(f"Warning: Could not delete existing log file: {e}")

        # Set up format for logging
        console_format = '%(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Set up console handler with configurable output level
        console_handler = logging.StreamHandler()
        if quiet:
            console_handler.setLevel(logging.CRITICAL)  # Only show critical errors in console
        else:
            console_handler.setLevel(logging.ERROR)     # Show errors and critical in console
        console_handler.setFormatter(logging.Formatter(console_format))

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Capture all levels for file logging

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        root_logger.addHandler(console_handler)

        # Set up file handler if log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Full detailed logging in file
            file_handler.setFormatter(logging.Formatter(file_format))
            root_logger.addHandler(file_handler)

        # Reduce logging from external libraries
        logging.getLogger('flickrapi').setLevel(logging.ERROR)
        logging.getLogger('requests').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)

    def write_results_log(self):
        """Write a detailed results log file with enhanced failure reporting"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.results_dir / 'processing_results.txt'

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("FLICKR TO IMMICH PROCESSING RESULTS\n")
                f.write("=================================\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Write summary counts
                f.write("SUMMARY\n-------\n")
                f.write(f"Total files processed: {self.stats['total_files']}\n")
                f.write(f"Successfully processed: {self.stats['successful']['count']}\n")
                f.write(f"Failed metadata: {self.stats['failed']['metadata']['count']}\n")
                f.write(f"Failed file copy: {self.stats['failed']['file_copy']['count']}\n")
                f.write(f"Skipped: {self.stats['skipped']['count']}\n")
                if 'partial_metadata' in self.stats:
                    f.write(f"Partial metadata success: {self.stats['partial_metadata']['count']}\n")
                f.write("\n")

                # Write successful files
                f.write("SUCCESSFUL FILES\n----------------\n")
                for source, dest, status in self.stats['successful']['details']:
                    f.write(f"Source: {source}\n")
                    f.write(f"Destination: {dest}\n")
                    f.write(f"Status: {status}\n")
                    f.write("-" * 50 + "\n")

                # Write partial metadata successes if any
                if 'partial_metadata' in self.stats and self.stats['partial_metadata']['files']:
                    f.write("\nPARTIAL METADATA SUCCESS\n----------------------\n")
                    f.write("These files were exported but only with basic metadata:\n")
                    for file in self.stats['partial_metadata']['files']:
                        f.write(f"File: {file}\n")
                    f.write("-" * 50 + "\n")

                # Write metadata failures
                f.write("\nMETADATA FAILURES\n-----------------\n")
                for file, error, exported in self.stats['failed']['metadata']['details']:
                    f.write(f"File: {file}\n")
                    f.write(f"Error: {error}\n")
                    f.write(f"File exported: {'Yes' if exported else 'No'}\n")
                    f.write("-" * 50 + "\n")

                # Write file copy failures
                f.write("\nFILE COPY FAILURES\n------------------\n")
                for file, error in self.stats['failed']['file_copy']['details']:
                    f.write(f"File: {file}\n")
                    f.write(f"Error: {error}\n")
                    f.write("-" * 50 + "\n")

                # Write skipped files
                f.write("\nSKIPPED FILES\n-------------\n")
                for file, reason in self.stats['skipped']['details']:
                    f.write(f"File: {file}\n")
                    f.write(f"Reason: {reason}\n")
                    f.write("-" * 50 + "\n")

            logging.info(f"Results log written to {results_file}")

        except Exception as e:
            logging.error(f"Error writing results log: {str(e)}")


    def _find_json_files(self, base_name: str) -> List[Path]:
        """
        Find JSON files matching various patterns:
        1. Single file without _part (e.g., 'albums.json')
        2. Single file with _part (e.g., 'albums_part001.json')
        3. Multiple part files (e.g., 'albums_part*.json')
        """
        # Debug: Show what we're looking for
        logging.info(f"Searching for {base_name} files")

        # Build all possible patterns
        patterns = [
            f"{base_name}.json",             # Case 1: Single file
            f"{base_name}_part*.json",       # Case 2 & 3: Part files
            f"{base_name}*.json"             # Case 4: Any file starting with base_name
        ]

        matching_files = []
        for pattern in patterns:
            files = list(self.metadata_dir.glob(pattern))
            if files:
                logging.info(f"Found {len(files)} files matching pattern {pattern}:")
                for f in files:
                    logging.info(f"  - {f.name}")
            matching_files.extend(files)

        # Remove duplicates and sort
        unique_files = sorted(set(matching_files))

        if not unique_files:
            logging.warning(f"No {base_name} files found with any pattern")
        else:
            logging.info(f"Found {len(unique_files)} unique {base_name} files")

        return unique_files

    def _load_multipart_json(self, base_name: str, key: str) -> Union[Dict, List]:
        """
        Load and merge JSON data from files with various naming patterns

        Args:
            base_name: Base name without extension (e.g., 'albums')
            key: The key in the JSON that contains the data we want

        Returns:
            Either a merged dictionary or list, depending on the data structure
        """
        data_files = self._find_json_files(base_name)

        if not data_files:
            logging.warning(f"No {base_name} files found")
            return {} if base_name != "albums" else []  # Return empty dict or list based on type

        try:
            # First file determines the data structure
            with open(data_files[0], 'r', encoding='utf-8') as f:
                first_file_data = json.load(f)
                if key not in first_file_data:
                    logging.warning(f"Key '{key}' not found in first file {data_files[0].name}")
                    return {} if base_name != "albums" else []

                # Initialize with appropriate type
                if isinstance(first_file_data[key], dict):
                    merged_data = {}
                else:  # List type
                    merged_data = []

            # Now process all files
            for data_file in data_files:
                logging.info(f"Processing {data_file.name}")
                with open(data_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if key in file_data:
                        if isinstance(merged_data, dict):
                            # Dictionary merge
                            merged_data.update(file_data[key])
                        else:
                            # List merge
                            merged_data.extend(file_data[key])
                        logging.info(f"Loaded {len(file_data[key])} entries from {data_file.name}")
                    else:
                        logging.warning(f"Key '{key}' not found in {data_file.name}")

            count = len(merged_data) if isinstance(merged_data, (dict, list)) else 1
            files_word = "files" if len(data_files) > 1 else "file"
            logging.info(f"Loaded total {count} entries from {len(data_files)} {base_name} {files_word}")
            return merged_data

        except Exception as e:
            logging.error(f"Error loading {base_name}: {str(e)}")
            return {} if base_name != "albums" else []

    def _get_user_info(self, user_id: str) -> Tuple[str, str]:
        """
        Get username and real name for a user ID using Flickr API
        Returns tuple of (username, realname)
        If API is unavailable or user not found, returns (user_id, "")
        """
        # Check cache first
        if user_id in self.user_info_cache:
            return self.user_info_cache[user_id]

        # Fall back to just user_id if no API available
        if not self.flickr:
            return (user_id, "")

        try:
            # Call Flickr API
            user_info = self.flickr.people.getInfo(api_key=os.environ['FLICKR_API_KEY'], user_id=user_id)

            # Parse response
            person = user_info.find('person')
            if person is None:
                raise ValueError("No person element found in response")

            username = person.find('username').text
            realname = person.find('realname')
            realname = realname.text if realname is not None else ""

            # Cache the result
            self.user_info_cache[user_id] = (username, realname)
            return (username, realname)

        except Exception as e:
            logging.warning(f"Failed to get user info for {user_id}: {e}")
            return (user_id, "")


    def _get_username(self, user_id: str) -> str:
        """Get username for a user ID, falling back to ID if not found"""
        username, _ = self._get_user_info(user_id)
        return username

    def _load_account_profile(self):
        """Load the account profile data"""
        profile_file = self.metadata_dir / 'account_profile.json'
        try:
            if not profile_file.exists():
                logging.warning("account_profile.json not found, some metadata will be missing")
                self.account_data = {}
                return

            with open(profile_file, 'r', encoding='utf-8') as f:
                self.account_data = json.load(f)
                logging.info(f"Loaded account profile for {self.account_data.get('real_name', 'unknown user')}")
        except Exception as e:
            logging.error(f"Error loading account profile: {str(e)}")
            self.account_data = {}

    def _process_user_mappings(self):
        """Process user mappings from loaded contacts data"""
        try:
            for username, url in self.contacts.items():
                # Extract user ID from URL, handling different URL formats
                if '/people/' in url:
                    user_id = url.split('/people/')[1].strip('/')
                else:
                    user_id = url.strip()

                # Store both the full ID and the cleaned version (without @N00 if present)
                self.user_mapping[user_id] = username
                if '@' in user_id:
                    clean_id = user_id.split('@')[0]
                    self.user_mapping[clean_id] = username

            logging.info(f"Processed {len(self.user_mapping)} user mappings")
            # Log first few mappings to help with debugging
            sample_mappings = dict(list(self.user_mapping.items())[:3])
            logging.info(f"Sample user mappings: {sample_mappings}")
        except Exception as e:
            logging.error(f"Error processing user mappings: {str(e)}")

    def _get_photo_favorites(self, photo_id: str) -> List[Dict]:
        """Fetch list of users who favorited a photo using Flickr API"""
        if not self.flickr:
            return []

        try:
            favorites = []
            page = 1
            per_page = 50

            while True:
                try:
                    # Get favorites for current page
                    response = self.flickr.photos.getFavorites(
                        api_key=os.environ['FLICKR_API_KEY'],
                        photo_id=photo_id,
                        page=page,
                        per_page=per_page
                    )

                    # Extract person elements
                    photo_elem = response.find('photo')
                    if photo_elem is None:
                        break

                    person_elems = photo_elem.findall('person')
                    if not person_elems:
                        break

                    # Process each person
                    for person in person_elems:
                        username = person.get('username', '')
                        nsid = person.get('nsid', '')
                        favedate = person.get('favedate', '')

                        # Convert favedate to readable format
                        if favedate:
                            try:
                                favedate = datetime.fromtimestamp(int(favedate)).strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                pass  # Keep original if conversion fails

                        favorites.append({
                            'username': username,
                            'nsid': nsid,
                            'favedate': favedate
                        })

                    # Check if we've processed all pages
                    total_pages = int(photo_elem.get('pages', '1'))
                    if page >= total_pages:
                        break

                    page += 1

                except Exception as e:
                    # If we encounter an error on a specific page, log it at debug level and break
                    logging.debug(f"Error fetching favorites page {page} for photo {photo_id}: {str(e)}")
                    break

            return favorites

        except Exception as e:
            # Log at debug level instead of warning since this is expected for some photos
            logging.debug(f"Failed to get favorites for photo {photo_id}: {str(e)}")
            return []

    def _fetch_user_interesting_photos(self, time_period: str, per_page: int = 100) -> List[Dict]:
        """Process user's photos and sort by engagement metrics."""
        try:
            photos = []
            photo_files = list(self.photos_dir.iterdir())

            logging.info(f"\nProcessing {len(photo_files)} local photos for engagement metrics")
            logging.info("Criteria: photos must have at least 1 favorite OR 1 comment")
            logging.info("Scoring: favorites × 10 + comments × 5")

            # Process each photo's metadata
            for photo_file in photo_files:
                if not photo_file.is_file() or not any(photo_file.name.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    continue

                # Extract photo ID from filename
                photo_id = None
                parts = photo_file.name.split('_')
                for i, part in enumerate(parts):
                    if i < len(parts) - 1 and part.isdigit():
                        photo_id = part
                        break

                if not photo_id:
                    continue

                # Load metadata using existing method
                photo_json_files = self._find_json_files(f"photo_{photo_id}")
                if not photo_json_files:
                    logging.warning(f"No metadata found for photo {photo_id}")
                    continue

                try:
                    with open(photo_json_files[0], 'r', encoding='utf-8') as f:
                        photo_metadata = json.load(f)
                except Exception as e:
                    logging.warning(f"Error reading metadata for photo {photo_id}: {e}")
                    continue

                try:
                    date_taken = datetime.strptime(photo_metadata.get('date_taken', ''), '%Y-%m-%d %H:%M:%S')
                    year_taken = date_taken.year
                except (ValueError, TypeError):
                    logging.warning(f"Invalid date for photo {photo_id}, skipping")
                    continue

                # Calculate engagement metrics
                faves = int(photo_metadata.get('count_faves', '0'))
                comments = int(photo_metadata.get('count_comments', '0'))
                views = int(photo_metadata.get('count_views', '0'))

                # Log each photo's metrics
                logging.info(f"\nPhoto {photo_id} engagement:")
                logging.info(f"  Title: {photo_metadata.get('name', '')}")
                logging.info(f"  Favorites: {faves}")
                logging.info(f"  Comments: {comments}")

                # Check if photo meets criteria and calculate score
                if faves > 10 or comments > 1 or views >= 20:  # Added views criterion
                    interestingness_score = (faves * 10) + (comments * 5) + (views * 0.1)  # Added views to score
                    logging.info(f"  Meets criteria! Score: {interestingness_score}")
                    logging.info(f"    Favorite points: {faves} × 10 = {faves * 10}")
                    logging.info(f"    Comment points: {comments} × 5 = {comments * 5}")
                    logging.info(f"    View points: {views} × 0.1 = {views * 0.1}")

                    photo_data = {
                        'id': photo_id,
                        'title': photo_metadata.get('name', ''),
                        'description': photo_metadata.get('description', ''),
                        'date_taken': photo_metadata.get('date_taken', ''),
                        'year_taken': year_taken,
                        'license': photo_metadata.get('license', ''),
                        'fave_count': faves,
                        'comment_count': comments,
                        'count_views': photo_metadata.get('count_views', '0'),
                        'count_faves': faves,
                        'count_comments': comments,
                        'interestingness_score': interestingness_score,
                        'original_file': photo_file,
                        'original': str(photo_file),
                        'geo': photo_metadata.get('geo', None),
                        'tags': photo_metadata.get('tags', []),
                        'photopage': photo_metadata.get('photopage', ''),
                        'privacy': photo_metadata.get('privacy', ''),
                        'safety': photo_metadata.get('safety', ''),
                        'comments': photo_metadata.get('comments', [])
                    }
                    photos.append(photo_data)
                else:
                    logging.info("  Does not meet minimum criteria (needs 1+ fave OR 1+ comment)")

            # Sort by interestingness score
            photos.sort(key=lambda x: x['interestingness_score'], reverse=True)

            # Log summary of findings
            if photos:
                scores = [p['interestingness_score'] for p in photos]
                logging.info("\nSummary:")
                logging.info(f"  Total photos scanned: {len(photo_files)}")
                logging.info(f"  Photos meeting criteria: {len(photos)}")
                logging.info(f"  Score range: {min(scores)} to {max(scores)}")
                logging.info(f"  Limiting to top {per_page} photos")

            # Limit to requested number
            photos = photos[:per_page]

            # Log the selected photos
            logging.info("\nSelected photos:")
            for i, photo in enumerate(photos, 1):
                logging.info(f"  {i}. {photo['title']} (ID: {photo['id']}):")
                logging.info(f"     Score: {photo['interestingness_score']}")
                logging.info(f"     Faves: {photo['fave_count']}, Comments: {photo['comment_count']}")

            return photos

        except Exception as e:
            logging.error(f"Error processing photos: {str(e)}")
            return []

    def create_interesting_albums(self, time_period: str, photo_count: int = 100):
        """Create albums of user's most engaging photos, organized by detailed privacy settings."""
        try:
            # Create highlights_only parent directory
            highlights_dir = self.output_dir / "highlights_only"
            highlights_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Processing your photos for engagement metrics...")

            # Get photos with engagement metrics
            all_photos = self._fetch_user_interesting_photos(time_period, photo_count)
            if not all_photos:
                logging.error("No photos found with engagement metrics")
                return

            # Initialize privacy groups dynamically
            privacy_groups = {}

            # Group photos by exact privacy setting with progress bar
            with tqdm(all_photos, desc="Analyzing privacy settings", unit="photo") as pbar:
                for photo in pbar:
                    privacy = photo.get('privacy', '').lower()
                    if not privacy:
                        privacy = 'private'  # Default to private if no setting

                    if privacy not in privacy_groups:
                        privacy_groups[privacy] = []
                    privacy_groups[privacy].append(photo)
                    pbar.set_postfix({'privacy': privacy})

            # Log summary of privacy groups found
            logging.info("\nPrivacy groups found:")
            for privacy, photos in privacy_groups.items():
                logging.info(f"- {privacy}: {len(photos)} photos")

            # Process each privacy group with an overall progress bar
            total_groups = len(privacy_groups)
            with tqdm(total=total_groups, desc="Processing privacy groups", unit="group") as group_pbar:
                for privacy, group in privacy_groups.items():
                    group_pbar.set_postfix({'current': privacy})

                    # Sort by interestingness score
                    group.sort(key=lambda x: x['interestingness_score'], reverse=True)
                    for i, photo in enumerate(group, 1):
                        photo['normalized_score'] = i

                    # Create folder name based on privacy setting
                    folder_name = privacy.replace(' ', '_').title() + '_Highlights'

                    if time_period == 'all-time':
                        self._create_single_interesting_album(
                            highlights_dir / folder_name,
                            f"All Time {folder_name}",
                            f"Your most engaging {privacy} Flickr photos across all time",
                            group
                        )
                    else:  # byyear
                        photos_by_year = {}
                        for photo in group:
                            year = photo['year_taken']
                            if year not in photos_by_year:
                                photos_by_year[year] = []
                            photos_by_year[year].append(photo)

                        # Process each year with its own progress bar
                        years = sorted(photos_by_year.keys(), reverse=True)
                        with tqdm(years, desc=f"Processing years for {privacy}", unit="year", leave=False) as year_pbar:
                            for year in year_pbar:
                                year_pbar.set_postfix({'year': year})
                                year_photos = photos_by_year[year]
                                if len(year_photos) > photo_count:
                                    year_photos = year_photos[:photo_count]

                                # Sort and rank within each year
                                year_photos.sort(key=lambda x: x['interestingness_score'], reverse=True)
                                for i, photo in enumerate(year_photos, 1):
                                    photo['normalized_score'] = i

                                self._create_single_interesting_album(
                                    highlights_dir / folder_name,
                                    f"{folder_name} {year}",
                                    f"Your most engaging {privacy} Flickr photos from {year}",
                                    year_photos
                                )

                    group_pbar.update(1)

            logging.info("\nFinished creating highlight folders:")
            for privacy in privacy_groups.keys():
                folder_name = privacy.replace(' ', '_').title() + '_Highlights'
                logging.info(f"- {folder_name}/")

        except Exception as e:
            logging.error(f"Error creating highlight albums: {str(e)}")
            raise

    def _create_single_interesting_album(self, album_dir: Path, album_name: str, description: str, photos: List[Dict]):
        """Create a single album of engaging photos using existing photo files"""
        try:
            album_dir.mkdir(parents=True, exist_ok=True)

            with tqdm(total=len(photos), desc=f"Processing {album_name}", leave=False) as pbar:
                for photo in photos:
                    try:
                        source_file = photo['original_file']
                        if not source_file.exists():
                            self.stats['skipped']['count'] += 1
                            self.stats['skipped']['details'].append(
                                (str(source_file), f"photo_{photo['id']}.json", "Source file missing for highlight")
                            )
                            pbar.update(1)
                            continue

                        # Create descriptive filename
                        safe_title = self._sanitize_folder_name(photo['title']) if photo['title'] else photo['id']
                        photo_filename = f"{safe_title}_{photo['id']}{Path(source_file).suffix}"
                        dest_file = album_dir / photo_filename

                        # Copy file
                        shutil.copy2(source_file, dest_file)

                        photo_for_metadata = photo.copy()
                        photo_for_metadata['original_file'] = str(source_file)
                        photo_for_metadata['original'] = str(source_file)

                        # Add engagement metrics to metadata
                        photo_for_metadata['engagement'] = {
                            'rank': photo['normalized_score'],
                            'total_ranked': len(photos),
                            'favorites': photo['fave_count'],
                            'comments': photo['comment_count']
                        }

                        # Ensure photopage exists
                        if 'photopage' not in photo_for_metadata:
                            photo_for_metadata['photopage'] = f"https://www.flickr.com/photos/{self.account_data.get('nsid', '')}/{photo['id']}"

                        # Embed metadata based on media type
                        media_type = self.get_media_type(dest_file)
                        if media_type == MediaType.IMAGE:
                            self._embed_image_metadata(dest_file, photo_for_metadata)
                        elif media_type == MediaType.VIDEO:
                            self._embed_video_metadata(dest_file, photo_for_metadata)

                        if self.write_xmp_sidecars:
                            self._write_xmp_sidecar(dest_file, photo_for_metadata)

                        self.stats['successful']['count'] += 1
                        pbar.update(1)

                    except Exception as e:
                        error_msg = f"Error processing highlight photo {photo['id']}: {str(e)}"
                        self.stats['failed']['count'] += 1
                        self.stats['failed']['details'].append(
                            (str(photo.get('original_file', f"unknown_{photo['id']}")),
                            f"photo_{photo['id']}.json",
                            error_msg)
                        )
                        logging.error(error_msg)
                        pbar.update(1)
                        continue

        except Exception as e:
            error_msg = f"Error creating album {album_name}: {str(e)}"
            self.stats['failed']['count'] += 1
            self.stats['failed']['details'].append(
                ("unknown", "unknown", error_msg)
            )
            logging.error(error_msg)
            raise

    def _validate_directories(self):
        """Validate input and output directories"""
        if not self.metadata_dir.exists():
            raise ValueError(f"Metadata directory does not exist: {self.metadata_dir}")

        if not self.photos_dir.exists():
            raise ValueError(f"Photos directory does not exist: {self.photos_dir}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def _extract_photo_id(self, filename: str) -> Optional[str]:
        """
        Extract photo ID from filename using multiple methods.
        Returns None if no ID can be found.
        """
        try:
            # Method 1: Look for ID between underscores
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if i < len(parts) - 1 and part.isdigit():
                    return part

            # Method 2: Look for number before extension
            base = os.path.splitext(filename)[0]
            parts = base.split('_')
            if parts[-1].isdigit():
                return parts[-1]

            # Method 3: Extract any number sequence from filename
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return numbers[0]

            return None
        except Exception as e:
            logging.warning(f"Error extracting photo ID from {filename}: {str(e)}")
            return None

    def _load_albums(self) -> dict:
        """Load and parse albums data (single or multi-part)"""
        try:
            # Try multi-part first
            albums_data = self._load_multipart_json("albums_part*.json", "albums")
            if albums_data:
                return albums_data

            # Fall back to single file if no multi-part files found
            albums_file = self.metadata_dir / 'albums.json'
            if not albums_file.exists():
                raise FileNotFoundError(f"No albums files found in {self.metadata_dir}")

            with open(albums_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'albums' not in data:
                    raise ValueError("Invalid albums.json format: 'albums' key not found")
                return data['albums']
        except Exception as e:
            raise ValueError(f"Error loading albums data: {str(e)}")

    def _build_photo_album_mapping(self):
        """Build mapping of photos to their albums"""
        try:
            for album in self.albums:
                if 'photos' not in album:
                    logging.warning(f"Album '{album.get('title', 'Unknown')}' has no photos key")
                    continue

                for photo_id in album['photos']:
                    if photo_id not in self.photo_to_albums:
                        self.photo_to_albums[photo_id] = []
                    self.photo_to_albums[photo_id].append(album['title'])

            logging.info(f"Found {len(self.photo_to_albums)} unique media items across albums")
        except Exception as e:
            raise ValueError(f"Error building photo-album mapping: {str(e)}")

    def get_media_type(self, file_path: Path) -> MediaType:
        """Determine the type of media file"""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('image/'):
                return MediaType.IMAGE
            elif mime_type.startswith('video/'):
                return MediaType.VIDEO
        return MediaType.UNKNOWN

    def create_album_structure(self):
        """Create the album folder structure"""
        try:
            # Create full_library_export/by_album directory
            full_export_dir = self.output_dir / "full_library_export" / "by_album"
            full_export_dir.mkdir(parents=True, exist_ok=True)

            # Explicitly create 00_NoAlbum directory first
            no_album_dir = full_export_dir / "00_NoAlbum"
            no_album_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Created 00_NoAlbum directory for unorganized photos")

            # Create album directories under by_album
            for album in self.albums:
                if album['title'] != '00_NoAlbum':  # Skip 00_NoAlbum as we already created it
                    album_dir = full_export_dir / self._sanitize_folder_name(album['title'])
                    album_dir.mkdir(parents=True, exist_ok=True)

            num_regular_albums = sum(1 for album in self.albums if album['title'] != '00_NoAlbum')
            logging.info(f"Created {num_regular_albums} regular album directories under by_album")

        except Exception as e:
            logging.error(f"Error creating album structure: {str(e)}")
            raise

    def create_date_structure(self, date_format: str):
        """Create the date-based folder structure"""
        try:
            # Create full_library_export/by_date directory
            date_export_dir = self.output_dir / "full_library_export" / "by_date"
            date_export_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Created date-based directory structure under by_date")
        except Exception as e:
            logging.error(f"Error creating date structure: {str(e)}")
            raise

    def _get_date_path(self, date_str: str, date_format: str) -> Path:
        """Convert date string to folder path based on format"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            if date_format == 'yyyy/yyyy-mm-dd':
                return Path(f"{date.year}/{date.year}-{date.month:02d}-{date.day:02d}")
            elif date_format == 'yyyy/yyyy-mm':
                return Path(f"{date.year}/{date.year}-{date.month:02d}")
            elif date_format == 'yyyy-mm-dd':
                return Path(f"{date.year}-{date.month:02d}-{date.day:02d}")
            else:
                raise ValueError(f"Unsupported date format: {date_format}")
        except Exception as e:
            logging.error(f"Error processing date {date_str}: {str(e)}")
            # Return a fallback path for invalid dates
            return Path("unknown_date")

    def _sanitize_folder_name(self, name: str) -> str:
        """Convert album name to safe folder name"""
        # Replace spaces with underscores and remove special characters
        sanitized = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                         for c in name.replace(' ', '_'))
        return sanitized.strip('_')

    def process_photos(self, organization: str, date_format: str = None):
        """Process all photos: copy to albums and embed metadata"""
        self.stats['total_files'] = len(self.photo_to_albums)

        # Create a thread pool for parallel processing
        max_workers = min(os.cpu_count() * 2, 8)  # Use up to 8 threads

        def process_photo_wrapper(args):
            """Wrapper function for thread pool"""
            photo_id, albums = args
            try:
                # First, find the source file to get the actual filename
                photo_json = self._load_photo_metadata(photo_id)
                source_file = self._find_photo_file(photo_id, photo_json['name'] if photo_json else photo_id)

                # Store the actual filename in the result tuple
                if organization == 'by_album':
                    result = self._process_single_photo_by_album(photo_id, albums)
                    return (result, photo_id, str(source_file) if source_file else f"unknown_{photo_id}")
                else:
                    result = self._process_single_photo_by_date(photo_id, date_format)
                    return (result, photo_id, str(source_file) if source_file else f"unknown_{photo_id}")
            except Exception as e:
                return (False, photo_id, f"unknown_{photo_id}", str(e))

        # Process photos in parallel
        items = list(self.photo_to_albums.items())
        with tqdm(total=len(items),
                desc="Processing photos",
                leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
            pbar.set_postfix_str("Starting...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_photo = {
                    executor.submit(process_photo_wrapper, item): item[0]
                    for item in items
                }

                for future in concurrent.futures.as_completed(future_to_photo):
                    photo_id = future_to_photo[future]
                    try:
                        result = future.result()
                        if isinstance(result, tuple):
                            if len(result) == 3:  # Success case
                                success, photo_id, filename = result
                                if success:
                                    self.stats['successful']['count'] += 1
                                # Update progress bar with actual filename
                                pbar.set_postfix_str(f"File: {os.path.basename(filename)}")
                            else:  # Error case
                                _, photo_id, filename, error = result
                                self.stats['failed']['count'] += 1
                                self.stats['failed']['details'].append(
                                    (filename, f"photo_{photo_id}.json", error)
                                )
                                pbar.set_postfix_str(f"File: {os.path.basename(filename)} (failed)")

                    except Exception as e:
                        self.stats['failed']['count'] += 1
                        self.stats['failed']['details'].append(
                            (f"unknown_{photo_id}", f"photo_{photo_id}.json", str(e))
                        )
                        pbar.set_postfix_str(f"File: unknown_{photo_id} (error)")
                    finally:
                        pbar.update(1)


    def _build_formatted_description(self, metadata: Dict) -> str:
        """Create a formatted description including key metadata fields based on configuration"""
        if not self.include_extended_description:
            return metadata.get("description", "")

        metadata_sections = []

        # Add highlight rank if available (always at the top)
        if 'engagement' in metadata and 'rank' in metadata['engagement'] and 'total_ranked' in metadata['engagement']:
            metadata_sections.extend([
                f"Highlight Rank: #{metadata['engagement']['rank']} (of {metadata['engagement']['total_ranked']})",
                "-----"  # Separator after rank
            ])

        # Add description if it exists and isn't empty
        if metadata.get("description"):
            metadata_sections.extend([
                "Description:",
                metadata.get("description")
            ])

        # Add comments section if there are any comments
        if metadata.get('comments'):
            metadata_sections.extend([
                "",
                "Flickr Comments:",
                *[f"- {self._format_user_comment(comment)}"
                for comment in metadata.get('comments', [])]
            ])

        # Add favorites section if API is connected and photo has favorites
        try:
            fave_count = int(metadata.get('count_faves', '0'))
        except (ValueError, TypeError):
            fave_count = 0

        if self.flickr and fave_count > 0:
            favorites = self._get_photo_favorites(metadata['id'])
            if favorites:
                metadata_sections.extend([
                    "",
                    "Flickr Faves:",
                    *[f"- {fave['username'] or fave['nsid']} ({fave['favedate']})"
                    for fave in favorites]
                ])

        # Add albums section
        if photo_id := metadata.get('id'):
            albums = self.photo_to_albums.get(photo_id, [])
            if albums:
                metadata_sections.extend([
                    "",
                    "Flickr Albums:",
                    *[f"- {album_name}" for album_name in albums]
                ])

        # Add the rest of metadata sections
        metadata_sections.extend([
            "",
            "-----",
            "Flickr Meta:",
            f"View Count: {metadata.get('count_views', '0')}",
            f"Favorite Count: {metadata.get('count_faves', '0')}",
            f"Comment Count: {metadata.get('count_comments', '0')}",
            "--",
            f"Privacy: {metadata.get('privacy', '')}",
            f"Safety Level: {metadata.get('safety', '')}",
            "--",
            f"Flickr URL: {metadata.get('photopage', '')}",
            f"Creator Profile: {self.account_data.get('screen_name', '')} / {self.account_data.get('profile_url', '')}",
            "--",
        ])

        # Filter out empty sections and join with newlines
        description = "\n".join(section for section in metadata_sections if section is not None and section != "")
        return description

    def _format_user_comment(self, comment) -> str:
        """Format a user comment with username and realname if available"""
        username, realname = self._get_user_info(comment['user'])
        if realname:
            user_display = f"{realname} ({username})"
        else:
            user_display = username
        return f"{user_display} ({comment['date']}): {comment['comment']}"

    def _process_single_photo_by_album(self, photo_id: str, album_names: List[str]) -> bool:
        """Process a single photo for album-based organization with enhanced error handling"""
        exported = False
        try:
            # Load photo metadata
            photo_json = self._load_photo_metadata(photo_id)
            if not photo_json:
                if self.block_if_failure:
                    self.stats['failed']['metadata']['count'] += 1
                    self.stats['failed']['metadata']['details'].append(
                        (f"photo_{photo_id}", "Metadata file not found", False)
                    )
                    return False
                else:
                    logging.warning(f"Metadata not found for {photo_id}, continuing with file copy only")

            # Find the source file
            source_file = self._find_photo_file(photo_id, photo_json['name'] if photo_json else photo_id)
            if not source_file:
                self.stats['failed']['file_copy']['count'] += 1
                self.stats['failed']['file_copy']['details'].append(
                    (f"photo_{photo_id}", "Source file not found")
                )
                return False

            # Process the photo for each album
            for album_name in album_names:
                try:
                    album_dir = self.output_dir / "full_library_export" / "by_album" / self._sanitize_folder_name(album_name)
                    album_dir.mkdir(parents=True, exist_ok=True)

                    # Determine destination filename with enhanced handling
                    if photo_json and photo_json.get('name'):
                        original_name = photo_json['name'].strip()
                        # Ensure name isn't empty after stripping
                        if not original_name:
                            original_name = f"photo_{photo_id}"
                    else:
                        original_name = f"photo_{photo_id}"

                    # Always ensure photo_id is in the filename
                    if photo_id not in original_name:
                        original_name = f"{original_name}_{photo_id}"

                    # Add extension if needed
                    source_extension = source_file.suffix.lower()
                    if not original_name.lower().endswith(source_extension):
                        original_name = f"{original_name}{source_extension}"

                    # Sanitize the filename
                    original_name = self._sanitize_filename(original_name)

                    dest_file = album_dir / original_name

                    # Skip if file exists and we're resuming
                    if self.resume and dest_file.exists():
                        self.stats['skipped']['count'] += 1
                        self.stats['skipped']['details'].append(
                            (str(source_file), "File already exists (resume mode)")
                        )
                        continue

                    # Copy file
                    try:
                        shutil.copy2(source_file, dest_file)
                        exported = True
                    except Exception as e:
                        self.stats['failed']['file_copy']['count'] += 1
                        self.stats['failed']['file_copy']['details'].append(
                            (str(source_file), f"Copy failed: {str(e)}")
                        )
                        if self.block_if_failure:
                            return False
                        continue

                    # Rest of the metadata processing...

                except Exception as e:
                    logging.error(f"Error processing {photo_id} in album {album_name}: {str(e)}")
                    if self.block_if_failure:
                        return False

            return exported

        except Exception as e:
            error_msg = f"Error processing {photo_id}: {str(e)}"
            self.stats['failed']['file_copy']['count'] += 1
            self.stats['failed']['file_copy']['details'].append(
                (str(source_file) if 'source_file' in locals() else f"photo_{photo_id}",
                error_msg)
            )
            logging.error(error_msg)
            return False

    def _process_single_photo_by_date(self, photo_id: str, date_format: str) -> bool:
        """Process a single photo for date-based organization with enhanced error handling"""
        exported = False
        try:
            # Load photo metadata
            photo_json = self._load_photo_metadata(photo_id)
            if not photo_json:
                if self.block_if_failure:
                    self.stats['failed']['metadata']['count'] += 1
                    self.stats['failed']['metadata']['details'].append(
                        (f"photo_{photo_id}", "Metadata file not found", False)
                    )
                    return False
                else:
                    logging.warning(f"Metadata not found for {photo_id}, continuing with file copy only")

            # Find the source file
            source_file = self._find_photo_file(photo_id, photo_json['name'] if photo_json else photo_id)
            if not source_file:
                self.stats['failed']['file_copy']['count'] += 1
                self.stats['failed']['file_copy']['details'].append(
                    (f"photo_{photo_id}", "Source file not found")
                )
                return False

            # Get date for directory structure
            if photo_json and 'date_taken' in photo_json:
                date_taken = photo_json['date_taken']
            else:
                try:
                    date_taken = datetime.fromtimestamp(source_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    logging.warning(f"Using file modification time for {photo_id} as date_taken is not available")
                except Exception as e:
                    if self.block_if_failure:
                        self.stats['failed']['metadata']['count'] += 1
                        self.stats['failed']['metadata']['details'].append(
                            (str(source_file), f"No date information available: {str(e)}", False)
                        )
                        return False
                    else:
                        date_taken = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        logging.warning(f"Using current date for {photo_id} as fallback")

            try:
                # Create date-based directory path
                date_path = self._get_date_path(date_taken, date_format)
                date_dir = self.output_dir / "full_library_export" / "by_date" / date_path
                date_dir.mkdir(parents=True, exist_ok=True)

                # Determine destination filename with enhanced handling
                if photo_json and photo_json.get('name'):
                    original_name = photo_json['name'].strip()
                    # Ensure name isn't empty after stripping
                    if not original_name:
                        original_name = f"photo_{photo_id}"
                else:
                    original_name = f"photo_{photo_id}"

                # Always ensure photo_id is in the filename
                if photo_id not in original_name:
                    original_name = f"{original_name}_{photo_id}"

                # Add extension if needed
                source_extension = source_file.suffix.lower()
                if not original_name.lower().endswith(source_extension):
                    original_name = f"{original_name}{source_extension}"

                # Sanitize the filename
                original_name = self._sanitize_filename(original_name)

                dest_file = date_dir / original_name

                # Skip if file exists and we're resuming
                if self.resume and dest_file.exists():
                    self.stats['skipped']['count'] += 1
                    self.stats['skipped']['details'].append(
                        (str(source_file), "File already exists (resume mode)")
                    )
                    return True

                # Copy file
                try:
                    shutil.copy2(source_file, dest_file)
                    exported = True
                except Exception as e:
                    self.stats['failed']['file_copy']['count'] += 1
                    self.stats['failed']['file_copy']['details'].append(
                        (str(source_file), f"Copy failed: {str(e)}")
                    )
                    if self.block_if_failure:
                        return False
                    return False

                # If we have metadata, try to embed it
                if photo_json:
                    try:
                        media_type = self.get_media_type(dest_file)

                        if media_type == MediaType.IMAGE:
                            self._embed_image_metadata(dest_file, photo_json)
                            if self.write_xmp_sidecars:
                                self._write_xmp_sidecar(dest_file, photo_json)
                        elif media_type == MediaType.VIDEO:
                            self._embed_video_metadata(dest_file, photo_json)
                            if self.write_xmp_sidecars:
                                self._write_xmp_sidecar(dest_file, photo_json)
                    except Exception as e:
                        self.stats['failed']['metadata']['count'] += 1
                        self.stats['failed']['metadata']['details'].append(
                            (str(dest_file), f"Metadata embedding failed: {str(e)}", True)
                        )
                        if self.block_if_failure:
                            # Remove the file if we're blocking on failures
                            dest_file.unlink(missing_ok=True)
                            return False

                if exported:
                    self.stats['successful']['count'] += 1
                    self.stats['successful']['details'].append(
                        (str(source_file), str(dest_file), "Exported" + (" with metadata" if photo_json else " without metadata"))
                    )
                    return True

                return False

            except Exception as e:
                error_msg = f"Error processing {photo_id}: {str(e)}"
                self.stats['failed']['file_copy']['count'] += 1
                self.stats['failed']['file_copy']['details'].append(
                    (str(source_file), error_msg)
                )
                logging.error(error_msg)
                if self.block_if_failure:
                    return False

                return exported

        except Exception as e:
            error_msg = f"Error processing {photo_id}: {str(e)}"
            self.stats['failed']['file_copy']['count'] += 1
            self.stats['failed']['file_copy']['details'].append(
                (str(source_file) if 'source_file' in locals() else f"photo_{photo_id}",
                error_msg)
            )
            logging.error(error_msg)
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to ensure it's valid and properly formatted
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Remove leading/trailing spaces and periods
        filename = filename.strip('. ')

        # Ensure filename isn't empty
        if not filename or filename.startswith('.'):
            return 'untitled_photo'

        # Limit filename length (max 255 chars is common filesystem limit)
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext

        return filename


    def _embed_image_metadata(self, photo_file: Path, metadata: Dict):
        """Embed metadata into an image file using exiftool with enhanced error handling"""
        try:
            # First, check if the file exists and is accessible
            if not photo_file.exists():
                raise FileNotFoundError(f"Source file does not exist: {photo_file}")

            # Clean up any leftover temporary files before starting
            temp_file = Path(str(photo_file) + "_exiftool_tmp")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logging.info(f"Cleaned up existing temporary file: {temp_file}")
                except Exception as e:
                    logging.warning(f"Could not remove existing temporary file {temp_file}: {e}")
                    return False

            # Build basic metadata arguments
            args = self._build_exiftool_args(photo_file, metadata)
            enhanced_description = self._build_formatted_description(metadata)

            # First verify JPEG integrity if it's a JPEG file
            if photo_file.suffix.lower() in ['.jpg', '.jpeg']:
                is_valid, error_msg = JPEGVerifier.is_jpeg_valid(str(photo_file))
                if not is_valid:
                    logging.warning(f"JPEG validation failed for {photo_file}: {error_msg}")
                    if JPEGVerifier.attempt_repair(str(photo_file)):
                        logging.info(f"Successfully repaired {photo_file}")
                    else:
                        # Try with minimal metadata as a last resort
                        try:
                            minimal_args = [
                                'exiftool',
                                '-overwrite_original',
                                '-ignoreMinorErrors',
                                '-m',
                                '-P',
                                '-fast',
                                '-safe',
                                f'-IPTC:Caption-Abstract={enhanced_description}',
                                str(photo_file)
                            ]
                            result = subprocess.run(minimal_args, capture_output=True, text=True)
                            if result.returncode == 0:
                                logging.info(f"Successfully embedded minimal metadata in {photo_file}")
                                # Add to stats that this was a partial success
                                if 'partial_metadata' not in self.stats:
                                    self.stats['partial_metadata'] = {'count': 0, 'files': []}
                                self.stats['partial_metadata']['count'] += 1
                                self.stats['partial_metadata']['files'].append(str(photo_file))
                                return True
                        except Exception as inner_e:
                            logging.error(f"Failed minimal metadata embed for {photo_file}: {str(inner_e)}")
                            self.stats['failed']['metadata']['count'] += 1
                            self.stats['failed']['metadata']['details'].append(
                                (str(photo_file), f"Failed minimal metadata embed: {str(inner_e)}", True)
                            )
                            return False

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # Clean up any temporary file before each attempt
                    if temp_file.exists():
                        temp_file.unlink()

                    # Try to embed metadata
                    result = subprocess.run(args, capture_output=True, text=True)

                    if result.returncode == 0:
                        if result.stderr:
                            logging.info(f"Metadata embedded with warnings for {photo_file}: {result.stderr}")
                        return True

                    if "Error renaming temporary file" in result.stderr:
                        # Wait briefly and retry
                        time.sleep(0.5)
                        retry_count += 1
                        continue

                    if "Format error" in result.stderr or "Not a valid JPG" in result.stderr:
                        # Try minimal embed
                        minimal_args = [
                            'exiftool',
                            '-overwrite_original',
                            '-ignoreMinorErrors',
                            '-m',
                            '-P',
                            '-fast',
                            '-safe',
                            f'-IPTC:Caption-Abstract={enhanced_description}',
                            str(photo_file)
                        ]
                        result = subprocess.run(minimal_args, capture_output=True, text=True)
                        if result.returncode == 0:
                            # Add to stats that this was a partial success
                            if 'partial_metadata' not in self.stats:
                                self.stats['partial_metadata'] = {'count': 0, 'files': []}
                            self.stats['partial_metadata']['count'] += 1
                            self.stats['partial_metadata']['files'].append(str(photo_file))
                            return True
                        break

                except Exception as e:
                    logging.error(f"Attempt {retry_count + 1} failed: {str(e)}")
                    retry_count += 1
                    time.sleep(0.5)

            if retry_count >= max_retries:
                error_msg = f"Failed to embed metadata after {max_retries} attempts for {photo_file}"
                logging.error(error_msg)
                self.stats['failed']['metadata']['count'] += 1
                self.stats['failed']['metadata']['details'].append(
                    (str(photo_file), error_msg, True)
                )
                return False

        except Exception as e:
            error_msg = f"Unexpected error embedding metadata in {photo_file}: {str(e)}"
            logging.error(error_msg)
            self.stats['failed']['metadata']['count'] += 1
            self.stats['failed']['metadata']['details'].append(
                (str(photo_file), error_msg, True)
            )
            return False

    def _load_photo_metadata(self, photo_id: str) -> Optional[Dict]:
        """Load metadata for a specific photo with enhanced file finding and logging"""
        try:
            # Try different possible metadata file patterns
            possible_patterns = [
                f"photo_{photo_id}.json",
                f"photo_{int(photo_id):d}.json",  # Handle numerical IDs
                f"photo_{photo_id.lstrip('0')}.json",  # Handle IDs with leading zeros
                f"{photo_id}.json",  # Try without 'photo_' prefix
                f"{int(photo_id):d}.json"  # Try without prefix and as integer
            ]

            # Debug log the search patterns
            logging.debug(f"Searching for metadata for photo {photo_id}")
            logging.debug(f"Looking in directory: {self.metadata_dir}")
            logging.debug(f"Trying patterns: {possible_patterns}")

            # Try each pattern
            for pattern in possible_patterns:
                photo_file = self.metadata_dir / pattern
                if photo_file.exists():
                    logging.debug(f"Found metadata file: {photo_file}")
                    with open(photo_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    logging.debug(f"Pattern {pattern} not found")

            # If we get here, we couldn't find the file
            # Do a directory listing to help debug
            logging.debug("Listing all JSON files in metadata directory:")
            json_files = list(self.metadata_dir.glob("*.json"))
            logging.debug(f"Found {len(json_files)} JSON files")
            if len(json_files) > 0:
                logging.debug("Sample of JSON files found:")
                for f in json_files[:5]:  # Show first 5 files
                    logging.debug(f"  - {f.name}")

            logging.error(f"Metadata file not found for photo {photo_id} after trying all patterns")
            return None

        except Exception as e:
            logging.error(f"Error loading metadata for photo {photo_id}: {str(e)}")
            return None

    def _find_photo_file(self, photo_id: str, filename: str) -> Optional[Path]:
        """Find the original photo file using the Flickr ID with enhanced logging"""
        logging.debug(f"Searching for photo file {photo_id} (filename: {filename})")
        logging.debug(f"Looking in directory: {self.photos_dir}")

        # First try: exact match with photo ID
        matches = []
        for file in self.photos_dir.iterdir():
            if f"_{photo_id}_" in file.name or f"_{photo_id}." in file.name:
                matches.append(file)
                logging.debug(f"Found exact match: {file.name}")

        if matches:
            if len(matches) > 1:
                logging.warning(f"Multiple matches found for {photo_id}, using first one: {matches[0]}")
            return matches[0]

        # Second try: normalize ID and try again
        normalized_id = photo_id.lstrip('0')  # Remove leading zeros
        logging.debug(f"Trying normalized ID: {normalized_id}")

        for file in self.photos_dir.iterdir():
            file_parts = file.name.split('_')
            for part in file_parts:
                clean_part = part.split('.')[0]  # Remove extension if present
                if clean_part.lstrip('0') == normalized_id:
                    logging.debug(f"Found match with normalized ID: {file.name}")
                    return file

        # Third try: look for filename
        clean_filename = filename.lower()
        for file in self.photos_dir.iterdir():
            if file.name.lower().startswith(clean_filename):
                logging.debug(f"Found match by filename: {file.name}")
                return file

        # If we get here, we couldn't find the file
        logging.debug("Listing sample of files in photos directory:")
        all_files = list(self.photos_dir.iterdir())
        logging.debug(f"Total files in directory: {len(all_files)}")
        if len(all_files) > 0:
            logging.debug("Sample of files found:")
            for f in all_files[:5]:  # Show first 5 files
                logging.debug(f"  - {f.name}")

        logging.error(f"Could not find media file for {photo_id} ({filename})")
        return None

    def _embed_image_metadata(self, photo_file: Path, metadata: Dict):
        """Embed metadata into an image file using exiftool"""
        try:
            args = self._build_exiftool_args(photo_file, metadata)
            result = subprocess.run(args, capture_output=True, text=True, check=True)

            if result.stderr:
                logging.warning(f"Exiftool warnings for {photo_file}: {result.stderr}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Error embedding metadata in {photo_file}: {e.stderr}"
            logging.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise

    def _embed_video_metadata(self, video_file: Path, metadata: Dict):
        """Embed metadata into a video file using exiftool"""
        try:
            # Build video-specific exiftool arguments
            args = self._build_exiftool_args(video_file, metadata, is_video=True)
            result = subprocess.run(args, capture_output=True, text=True, check=True)

            if result.stderr:
                logging.warning(f"Exiftool warnings for {video_file}: {result.stderr}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Error embedding metadata in {video_file}: {e.stderr}"
            logging.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise

    def _write_xmp_sidecar(self, media_file: Path, metadata: Dict):
        """Create XMP sidecar file with extended and Flickr-specific metadata"""
        sidecar_file = Path(str(media_file) + '.xmp')

        # Get enhanced description using existing method
        enhanced_description = self._build_formatted_description(metadata)

        # Build tag list from Flickr tags
        tags = [tag["tag"] for tag in metadata.get('tags', [])]

        # Function to safely encode text for XML
        def xml_escape(text):
            if not isinstance(text, str):
                text = str(text)
            return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

        xmp_content = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.1.2">
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:dc="http://purl.org/dc/elements/1.1/"
            xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
            xmlns:xmp="http://ns.adobe.com/xap/1.0/"
            xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
            xmlns:flickr="http://flickr.com/schema/2024/01/">
    <rdf:Description rdf:about="">
        <!-- Enhanced Description -->
        <dc:description>
            <rdf:Alt>
                <rdf:li xml:lang="x-default">{xml_escape(enhanced_description)}</rdf:li>
            </rdf:Alt>
        </dc:description>
        <!-- Engagement Metrics -->
                <flickr:engagement rdf:parseType="Resource">
                    <flickr:rank>{xml_escape(str(metadata.get('engagement', {}).get('rank', '0')))}</flickr:rank>
                    <flickr:totalRanked>{xml_escape(str(metadata.get('engagement', {}).get('total_ranked', '0')))}</flickr:totalRanked>
                    <flickr:favoriteCount>{xml_escape(str(metadata.get('engagement', {}).get('favorites', '0')))}</flickr:favoriteCount>
                    <flickr:commentCount>{xml_escape(str(metadata.get('engagement', {}).get('comments', '0')))}</flickr:commentCount>
                </flickr:engagement>
        <!-- Tags -->
        <dc:subject>
            <rdf:Bag>
                {''.join(f'<rdf:li>{xml_escape(tag)}</rdf:li>' for tag in tags)}
            </rdf:Bag>
        </dc:subject>

        <!-- Basic Metadata -->
        <dc:title>
            <rdf:Alt>
                <rdf:li xml:lang="x-default">{xml_escape(metadata.get("name", ""))}</rdf:li>
            </rdf:Alt>
        </dc:title>
        <dc:creator>
            <rdf:Seq>
                <rdf:li>{xml_escape(self.account_data.get("real_name", ""))}</rdf:li>
            </rdf:Seq>
        </dc:creator>
        <dc:rights>
            <rdf:Alt>
                <rdf:li xml:lang="x-default">{xml_escape(metadata.get("license", "All Rights Reserved"))}</rdf:li>
            </rdf:Alt>
        </dc:rights>
        <xmp:CreateDate>{xml_escape(metadata["date_taken"])}</xmp:CreateDate>
        <xmp:ModifyDate>{xml_escape(metadata["date_taken"])}</xmp:ModifyDate>

        <!-- Photo-specific Flickr metadata -->
        <flickr:id>{xml_escape(metadata["id"])}</flickr:id>
        <flickr:photopage>{xml_escape(metadata["photopage"])}</flickr:photopage>
        <flickr:original>{xml_escape(metadata["original"])}</flickr:original>
        <flickr:viewCount>{xml_escape(metadata.get("count_views", "0"))}</flickr:viewCount>
        <flickr:favoriteCount>{xml_escape(metadata.get("count_faves", "0"))}</flickr:favoriteCount>
        <flickr:commentCount>{xml_escape(metadata.get("count_comments", "0"))}</flickr:commentCount>
        <flickr:tagCount>{xml_escape(metadata.get("count_tags", "0"))}</flickr:tagCount>
        <flickr:noteCount>{xml_escape(metadata.get("count_notes", "0"))}</flickr:noteCount>

        <!-- Privacy and Permissions -->
        <flickr:privacy>{xml_escape(metadata.get("privacy", ""))}</flickr:privacy>
        <flickr:commentPermissions>{xml_escape(metadata.get("comment_permissions", ""))}</flickr:commentPermissions>
        <flickr:taggingPermissions>{xml_escape(metadata.get("tagging_permissions", ""))}</flickr:taggingPermissions>
        <flickr:safety>{xml_escape(metadata.get("safety", ""))}</flickr:safety>

        <!-- Account Information -->
        <flickr:accountInfo rdf:parseType="Resource">
            <flickr:realName>{xml_escape(self.account_data.get("real_name", ""))}</flickr:realName>
            <flickr:screenName>{xml_escape(self.account_data.get("screen_name", ""))}</flickr:screenName>
            <flickr:joinDate>{xml_escape(self.account_data.get("join_date", ""))}</flickr:joinDate>
            <flickr:profileUrl>{xml_escape(self.account_data.get("profile_url", ""))}</flickr:profileUrl>
            <flickr:nsid>{xml_escape(self.account_data.get("nsid", ""))}</flickr:nsid>
            <flickr:proUser>{xml_escape(self.account_data.get("pro_user", "no"))}</flickr:proUser>
        </flickr:accountInfo>

        <!-- Comments -->
        <flickr:comments>
            <rdf:Bag>
                {''.join(f'''<rdf:li rdf:parseType="Resource">
                <flickr:commentId>{xml_escape(comment["id"])}</flickr:commentId>
                <flickr:commentDate>{xml_escape(comment["date"])}</flickr:commentDate>
                <flickr:commentUser>{xml_escape(comment["user"])}</flickr:commentUser>
                <flickr:commentText>{xml_escape(comment["comment"])}</flickr:commentText>
                </rdf:li>''' for comment in metadata.get("comments", []))}
            </rdf:Bag>
        </flickr:comments>

        <!-- Favorites -->
        <flickr:favorites>
            <rdf:Bag>
                {''.join(f'''<rdf:li rdf:parseType="Resource">
                <flickr:favoriteUser>{xml_escape(fave["username"] or fave["nsid"])}</flickr:favoriteUser>
                <flickr:favoriteDate>{xml_escape(fave["favedate"])}</flickr:favoriteDate>
                </rdf:li>''' for fave in (self._get_photo_favorites(metadata['id']) if self.flickr and int(metadata.get('count_faves', '0')) > 0 else []))}
            </rdf:Bag>
        </flickr:favorites>

        <!-- Albums -->
        <flickr:albums>
            <rdf:Bag>
                {''.join(f'''<rdf:li>{xml_escape(album_name)}</rdf:li>'''
                for album_name in self.photo_to_albums.get(metadata.get('id', ''), []))}
            </rdf:Bag>
        </flickr:albums>

    </rdf:Description>
    </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end="w"?>"""

        with open(sidecar_file, 'w', encoding='utf-8') as f:
            f.write(xmp_content)

    def _build_gps_xmp(self, geo: Dict) -> str:
        """Build GPS XMP tags if geo data is available"""
        if not geo or 'latitude' not in geo or 'longitude' not in geo:
            return ""

        return f"""
        <exif:GPSLatitude>{geo['latitude']}</exif:GPSLatitude>
        <exif:GPSLongitude>{geo['longitude']}</exif:GPSLongitude>"""

    def _build_exiftool_args(self, media_file: Path, metadata: Dict, is_video: bool = False) -> List[str]:
        """Build exiftool arguments for metadata embedding - standard metadata only"""
        enhanced_description = self._build_formatted_description(metadata)
        args = [
            'exiftool',
            '-overwrite_original',
            '-ignoreMinorErrors',
            '-m',

            # Core timestamp metadata
            f'-DateTimeOriginal={metadata["date_taken"]}',
            f'-CreateDate={metadata["date_taken"]}',

            # Basic descriptive metadata
            f'-Title={metadata.get("name", "")}',
            f'-ImageDescription={enhanced_description}',
            f'-IPTC:Caption-Abstract={enhanced_description}',
            f'-Copyright={metadata.get("license", "All Rights Reserved")}',
            f'-Artist={self.account_data.get("real_name", "")}',
            f'-Creator={self.account_data.get("real_name", "")}',

            # Basic tags
            *[f'-Keywords={tag["tag"]}' for tag in metadata.get('tags', [])],
        ]

        # Handle image orientation using PIL for verification
        if media_file.suffix.lower() in ['.jpg', '.jpeg', '.tiff', '.tif']:
            try:
                with Image.open(media_file) as img:
                    # Get existing EXIF data
                    exif = img._getexif()
                    if exif:
                        # Find the orientation tag
                        orientation_tag = None
                        for tag_id in ExifTags.TAGS:
                            if ExifTags.TAGS[tag_id] == 'Orientation':
                                orientation_tag = tag_id
                                break

                        current_orientation = exif.get(orientation_tag, 1)
                        logging.debug(f"Current image orientation: {current_orientation}")

                        # If we have rotation metadata from Flickr
                        if 'rotation' in metadata:
                            rotation_degrees = int(metadata["rotation"])
                            # Map Flickr's rotation degrees to EXIF orientation
                            # Note: Flickr uses CCW rotation, EXIF uses CW
                            rotation_map = {
                                0: 1,    # Normal
                                90: 8,   # Rotate 270 CW (90 CCW)
                                180: 3,  # Rotate 180
                                270: 6   # Rotate 90 CW (270 CCW)
                            }
                            new_orientation = rotation_map.get(rotation_degrees, 1)

                            # Set orientation in EXIF
                            args.extend([
                                f'-IFD0:Orientation#={new_orientation}',
                                '-IFD0:YCbCrPositioning=1',  # Ensure proper color space positioning
                                '-IFD0:YCbCrSubSampling=2 2'  # Standard chroma subsampling
                            ])
                            logging.debug(f"Setting new orientation: {new_orientation} for rotation {rotation_degrees}")
            except Exception as e:
                logging.warning(f"Error checking image orientation: {str(e)}")

        # Standard GPS data if available
        if metadata.get('geo'):
            geo = metadata['geo']
            if 'latitude' in geo and 'longitude' in geo:
                args.extend([
                    f'-GPSLatitude={geo["latitude"]}',
                    f'-GPSLongitude={geo["longitude"]}',
                ])

        # Add the media file at the end
        args.append(str(media_file))

        return args

    def print_statistics(self):
        """Print processing statistics"""
        logging.info("\nProcessing Statistics:")
        logging.info(f"Total files: {self.stats['total_files']}")
        logging.info(f"Successfully processed: {self.stats['successful']['count']}")
        logging.info(f"Failed: {self.stats['failed']['count']}")
        logging.info(f"Skipped: {self.stats['skipped']['count']}")

        if self.stats['failed']['count'] > 0:
            logging.info("\nFailed files summary:")
            for img_file, json_file, error in self.stats['failed']['details'][:5]:  # Show first 5 failures
                logging.info(f"- {img_file}: {error}")
            if len(self.stats['failed']['details']) > 5:
                logging.info(f"... and {len(self.stats['failed']['details']) - 5} more failures")
                logging.info("See processing_results.txt for complete details")

def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Convert Flickr export to Immich-compatible format')
    parser.add_argument('--metadata-dir', required=True, help='Directory containing Flickr JSON metadata files')
    parser.add_argument('--photos-dir', required=True, help='Directory containing the photos/videos')
    parser.add_argument('--output-dir', required=True, help='Directory where album structure will be created')
    parser.add_argument('--log-file', default='flickr_to_immich.log', help='Log file location (default: flickr_to_immich.log)')
    parser.add_argument('--results-dir', help='Directory to store the processing results log (default: same as output directory)')
    parser.add_argument('--no-extended-description', action='store_true', help='Only include original description, skip additional metadata')
    parser.add_argument('--no-xmp-sidecars', action='store_true', help='Skip writing XMP sidecar files')
    parser.add_argument('--export-block-if-failure', action='store_true',
                       help='Block file export if metadata processing fails (default: export files even if metadata fails)')
    parser.add_argument('--resume', action='store_true', help='Resume previous run - skip existing files')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce console output (errors will still be logged to file)')


    # Export type flags
    export_type = parser.add_mutually_exclusive_group()
    export_type.add_argument('--export-interesting-only', action='store_true',
                           help='Only export the highlights/interesting photos')
    export_type.add_argument('--export-standard-only', action='store_true',
                           help='Only export the standard library (by album or by date)')

    # Interesting photos configuration
    parser.add_argument('--interesting-period', choices=['all-time', 'byyear'],
                       help='Time period for interesting photos: all-time or byyear')
    parser.add_argument('--interesting-count', type=int, default=100,
                       help='Number of interesting photos to fetch (max 500)')

    # Organization configuration
    parser.add_argument('--organization', choices=['by_album', 'by_date'], default='by_album',
                       help='How to organize photos in the library: by_album or by_date')
    parser.add_argument('--date-format', choices=['yyyy/yyyy-mm-dd', 'yyyy/yyyy-mm', 'yyyy-mm-dd'],
                       default='yyyy/yyyy-mm-dd',
                       help='Date format for folder structure when using by_date organization')

    api_key = os.environ.get('FLICKR_API_KEY')
    args = parser.parse_args()

    try:
        # Initialize converter with resume flag
        converter = FlickrToImmich(
            metadata_dir=args.metadata_dir,
            photos_dir=args.photos_dir,
            output_dir=args.output_dir,
            log_file=args.log_file,
            results_dir=args.results_dir,
            api_key=api_key,
            include_extended_description=not args.no_extended_description,
            write_xmp_sidecars=not args.no_xmp_sidecars,
            block_if_failure=args.export_block_if_failure,
            resume=args.resume,
            quiet=args.quiet
        )

        # Determine what to export based on flags
        export_standard = not args.export_interesting_only  # Export standard unless interesting-only flag is set
        export_interesting = (not args.export_standard_only and args.interesting_period)  # Export interesting if period is set and not standard-only

        # Create standard export (by_album or by_date) if needed
        if export_standard:
            logging.info("Creating standard library structure...")
            if args.organization == 'by_album':
                converter.create_album_structure()
            else:  # by_date
                converter.create_date_structure(args.date_format)

            # Process all media files
            logging.info("Processing media files for standard export...")
            converter.process_photos(args.organization, args.date_format)

        # Create interesting/highlights export if needed
        if export_interesting:
            logging.info(f"Creating album(s) of interesting photos for {args.interesting_period}...")
            converter.create_interesting_albums(
                args.interesting_period,
                args.interesting_count
            )

        # Print statistics
        converter.print_statistics()

        # Write detailed results log
        converter.write_results_log()

        # Print summary of what was exported
        logging.info("\nExport Summary:")
        if export_standard:
            if args.organization == 'by_album':
                logging.info(f"- Standard export (by album): {args.output_dir}/full_library_export/by_album")
            else:
                logging.info(f"- Standard export (by date): {args.output_dir}/full_library_export/by_date")
        if export_interesting:
            logging.info(f"- Highlights export: {args.output_dir}/highlights_only")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
