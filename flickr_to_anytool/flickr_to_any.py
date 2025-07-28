"""
**** FLICKR to ANY TOOL ****
by Rob Brown
https://github.com/brownphotographic/Flickr2Any-Tool

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
from tqdm import tqdm
import os
import io
import wx
import flickrapi
import xml.etree.ElementTree as ET
from importlib import metadata
from tomlkit.api import key
from PIL import Image
import concurrent.futures
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import ExifTags
from gooey import Gooey, GooeyParser
import argparse
import sys
import traceback
from tqdm import tqdm
import time
import psutil
import gc
import re


# Configure environment for progress bars and disable most logging
os.environ['TQDM_DISABLE'] = 'false'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Disable all logging except critical
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('flickrapi').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.CRITICAL)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory = process.memory_info()
    logging.info(f"Memory usage: {memory.rss / (1024 * 1024):.1f} MB")

def is_cli_mode():
    return '--cli' in sys.argv

class MediaType(Enum):
    """Supported media types"""
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"

# Configuration flags
INCLUDE_EXTENDED_DESCRIPTION = True  # Set to False to only include original description
WRITE_XMP_SIDECARS = True  # Set to False to skip writing XMP sidecar files

EXIF_ORIENTATION_MAP = {
    1: {
        'text': 'top, left',
        'description': 'Horizontal (normal)',
        'rotation': 0,
        'mirrored': False
    },
    2: {
        'text': 'top, right',
        'description': 'Mirror horizontal',
        'rotation': 0,
        'mirrored': True
    },
    3: {
        'text': 'bottom, right',
        'description': 'Rotate 180',
        'rotation': 180,
        'mirrored': False
    },
    4: {
        'text': 'bottom, left',
        'description': 'Mirror vertical',
        'rotation': 180,
        'mirrored': True
    },
    5: {
        'text': 'left, right',
        'description': 'Mirror horizontal and rotate 270 CW',
        'rotation': 270,
        'mirrored': True
    },
    6: {
        'text': 'right, top',
        'description': 'Rotate 90 CW',
        'rotation': 90,
        'mirrored': False
    },
    7: {
        'text': 'right, bottom',
        'description': 'Mirror horizontal and rotate 90 CW',
        'rotation': 90,
        'mirrored': True
    },
    8: {
        'text': 'left, top',
        'description': 'Rotate 270 CW',
        'rotation': 270,
        'mirrored': False
    }
}

def setup_directory_widgets(preprocessing, main_settings):
    """Add directory widgets to existing argument groups"""

    # Add source directory widget
    source_dir = preprocessing.add_argument(
        '--source-dir',
        metavar='Source Directory',
        widget='DirChooser',
        help='Directory for Flickr zip files',
        gooey_options={'full_width': True}
    )

    # Add metadata directory widget
    metadata_dir = main_settings.add_argument(
        '--metadata-dir',
        metavar='Metadata Directory',
        widget='DirChooser',
        help='Directory for metadata files',
        gooey_options={'full_width': True}
    )

    # Add photos directory widget
    photos_dir = main_settings.add_argument(
        '--photos-dir',
        metavar='Photos Directory',
        widget='DirChooser',
        help='Directory for photos',
        gooey_options={'full_width': True}
    )

    # Add output directory widget
    output_dir = main_settings.add_argument(
        '--output-dir',
        metavar='Output Directory',
        widget='DirChooser',
        help='Directory for output files',
        gooey_options={'full_width': True}
    )

    # Add results directory widget
    results_dir = main_settings.add_argument(
        '--results-dir',
        metavar='Results Directory',
        widget='DirChooser',
        help='Directory for results files',
        gooey_options={'full_width': True}
    )

    return preprocessing, main_settings

class FlickrPreprocessor:
    """Handles preprocessing of Flickr export zip files"""

    def __init__(self, source_dir: str, metadata_dir: str, photos_dir: str, quiet: bool = False):
        """
        Initialize the preprocessor

        Args:
            source_dir: Directory containing Flickr export zip files
            metadata_dir: Directory where metadata files should be extracted
            photos_dir: Directory where photos/videos should be extracted
            quiet: Whether to reduce console output
        """
        self.source_dir = Path(source_dir)
        self.metadata_dir = Path(metadata_dir)
        self.photos_dir = Path(photos_dir)
        self.quiet = quiet

        # Initialize statistics
        self.stats = {
            'metadata_files_processed': 0,
            'media_files_processed': 0,
            'errors': [],
            'skipped_files': []
        }

    def _clear_directory(self, directory: Path):
        """Clear all contents of a directory"""
        logging.debug(f"Clearing directory: {directory}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        logging.debug(f"Directory cleared and recreated: {directory}")

    def _prepare_directories(self):
        """Prepare directories by clearing them and ensuring they exist"""
        self._clear_directory(self.metadata_dir)
        self._clear_directory(self.photos_dir)
        logging.debug("Directories prepared for preprocessing")

    def _is_metadata_zip(self, filename: str) -> bool:
        """Check if a file is a metadata zip file"""
        # Check if filename matches pattern: numbers_alphanumeric_partN.zip
        import re
        pattern = r'\d+_[a-zA-Z0-9]+_part\d+\.zip$'
        return bool(re.match(pattern, filename))

    def _is_media_zip(self, filename: str) -> bool:
        """Check if a file is a media zip file"""
        import re
        patterns = [
            r'^data-download-\d+\.zip$',  # Pattern for photo zip files
            r'^data_\d+_[a-f0-9]+_\d+\.zip$'  # Pattern for alternate format
        ]
        return any(re.match(pattern, filename) for pattern in patterns)

    def _extract_zip(self, zip_path: Path, extract_path: Path, desc: str) -> tuple[int, list[str]]:
        """
        Extract a zip file with progress tracking

        Returns:
            Tuple of (files_processed, error_list)
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract
                files = zip_ref.namelist()
                total_files = len(files)
                processed = 0
                errors = []

                # Extract files with status updates
                print(f"Extracting {desc}")  # Initial message
                for i, file in enumerate(files, 1):
                    try:
                        zip_ref.extract(file, extract_path)
                        processed += 1
                        # Show progress with current filename (truncated if too long)
                        filename = os.path.basename(file)
                        if len(filename) > 30:  # Truncate long filenames
                            filename = filename[:27] + "..."
                        print(f"\rExtracting: {i}/{total_files} ({(i/total_files)*100:.1f}%) - {filename}",
                              end='', flush=True)
                    except Exception as e:
                        errors.append(f"Error extracting {file} from {zip_path.name}: {str(e)}")

                print()  # Final newline
                print(f"Completed: {processed} files extracted")  # Final status

                return processed, errors

        except Exception as e:
            return 0, [f"Error processing {zip_path.name}: {str(e)}"]

    def _process_zip_file(self, zip_path: Path) -> tuple[int, list[str]]:
        """Process a single zip file"""
        if self._is_metadata_zip(zip_path.name):
            return self._extract_zip(zip_path, self.metadata_dir, f"metadata from {zip_path.name}")
        elif self._is_media_zip(zip_path.name):
            return self._extract_zip(zip_path, self.photos_dir, f"media from {zip_path.name}")
        else:
            return 0, [f"Skipped unknown zip file: {zip_path.name}"]

    def process_exports(self):
        """Process all export files in the source directory"""
        try:
            # First clear and prepare directories
            self._prepare_directories()

            # Get list of zip files
            zip_files = [f for f in self.source_dir.iterdir() if f.suffix.lower() == '.zip']
            if not zip_files:
                raise ValueError(f"No zip files found in {self.source_dir}")

            logging.debug(f"Found {len(zip_files)} zip files to process")

            # Process files using thread pool
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
                # Create future to zip_path mapping
                future_to_zip = {executor.submit(self._process_zip_file, zip_path): zip_path
                               for zip_path in zip_files}

                # Process results as they complete
                for future in tqdm(as_completed(future_to_zip),
                                 total=len(zip_files),
                                 desc="Processing zip files",
                                 unit="files",
                                 disable=False,
                                 file=sys.stdout):
                    zip_path = future_to_zip[future]
                    try:
                        processed, errors = future.result()

                        # Update statistics
                        if self._is_metadata_zip(zip_path.name):
                            self.stats['metadata_files_processed'] += processed
                        elif self._is_media_zip(zip_path.name):
                            self.stats['media_files_processed'] += processed

                        # Record any errors
                        if errors:
                            self.stats['errors'].extend(errors)

                    except Exception as e:
                        self.stats['errors'].append(f"Error processing {zip_path.name}: {str(e)}")

            # Only print statistics if not in quiet mode
            if not self.quiet:
                self._print_statistics()

        except Exception as e:
            logging.error(f"Error during export processing: {str(e)}")
            raise

    def _print_statistics(self):
        """Print processing statistics"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)

        # Store and replace handlers temporarily
        original_handlers = logging.getLogger().handlers
        logging.getLogger().handlers = [console_handler]

        try:
            logging.info("\nPreprocessing Results")
            logging.info("-------------------")
            logging.info(f"Metadata files: {self.stats['metadata_files_processed']}")
            logging.info(f"Media files: {self.stats['media_files_processed']}")

            if self.stats['errors']:
                logging.error("\nErrors encountered:")
                for error in self.stats['errors']:
                    logging.error(f"- {error}")

            if self.stats['skipped_files']:
                logging.debug("\nSkipped files:")
                for skipped in self.stats['skipped_files']:
                    logging.debug(f"- {skipped}")
        finally:
            # Restore original handlers
            logging.getLogger().handlers = original_handlers

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

                # Read file and look for EOI marker
                buffer_size = 1024 * 1024  # 1MB buffer
                last_byte = None

                while True:
                    buffer = f.read(buffer_size)
                    if not buffer:
                        return False, "Missing JPEG EOI marker"

                    # Check for EOI marker split across reads
                    if last_byte == b'\xFF' and buffer.startswith(b'\xD9'):
                        return True, ""

                    # Look for EOI marker within buffer
                    if b'\xFF\xD9' in buffer:
                        return True, ""

                    # Save last byte for next iteration
                    last_byte = buffer[-1:]

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

    logging.getLogger().setLevel(logging.WARNING)

    # Modified SUPPORTED_EXTENSIONS initialization
    SUPPORTED_EXTENSIONS = {
        ext.lower() for ext in {
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.webp',
            '.JPG', '.JPEG', '.PNG', '.GIF', '.TIFF', '.TIF', '.WEBP',
            # Videos
            '.mp4', '.mov', '.avi', '.mpg', '.mpeg', '.m4v', '.3gp', '.wmv',
            '.webm', '.mkv', '.flv',
            '.MP4', '.MOV', '.AVI', '.MPG', '.MPEG', '.M4V', '.3GP', '.WMV',
            '.WEBM', '.MKV', '.FLV'
        }
    }

    def _is_supported_extension(self, filename: str) -> bool:
        """Check if a file has a supported extension, case-insensitive"""
        return any(filename.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def _clear_output_directory(self):
        """Clear all contents of a directory"""
        try:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True)
            logging.debug(f"Directory cleared and recreated: {self.output_dir}")
        except Exception as e:
            raise ValueError(f"Error clearing output directory: {str(e)}")

    def _get_destination_filename(self, photo_id: str, source_file: Path, photo_json: Optional[Dict]) -> str:
        """Get the destination filename for a photo, without Flickr ID"""
        try:
            # Get original name from metadata if available
            if photo_json and photo_json.get('name'):
                original_name = photo_json['name'].strip()
                # Ensure name isn't empty after stripping
                if not original_name:
                    original_name = source_file.name
            else:
                # Use the original filename
                original_name = source_file.name

            # Remove all Flickr IDs from filename
            # Split name and extension
            name_base, ext = os.path.splitext(original_name)

            # Remove any trailing Flickr ID patterns
            # Common patterns: name_123456789 or name_123456789_o
            import re
            # Remove ID patterns like _123456789 or _123456789_o
            name_base = re.sub(r'_\d{8,11}(?:_o)?$', '', name_base)

            # In case there are multiple IDs, keep removing them
            while re.search(r'_\d{8,11}(?:_o)?', name_base):
                name_base = re.sub(r'_\d{8,11}(?:_o)?', '', name_base)

            # Reconstruct filename
            dest_filename = f"{name_base}{ext}"

            # Ensure extension is present
            if not dest_filename.lower().endswith(source_file.suffix.lower()):
                dest_filename = f"{dest_filename}{source_file.suffix}"

            # Sanitize the filename
            return self._sanitize_filename(dest_filename)

        except Exception as e:
            logging.error(f"Error creating destination filename for {photo_id}: {str(e)}")
            # Fallback to a safe filename without ID
            name_base = source_file.stem
            # Remove any Flickr IDs from the fallback name
            name_base = re.sub(r'_\d{8,11}(?:_o)?', '', name_base)
            return f"{name_base}{source_file.suffix}"

    # THIS SECTION ENABLES ADDITIONAL METADATA TO BE ACCESSED FROM API IN PLACE OF THE JSON FILES, WHICH ARE UNRELIABLE
    def _get_metadata_from_api(self, photo_id: str) -> Optional[Dict]:
        """Get metadata from Flickr API with JSON fallback"""
        logging.debug(f"\nAttempting API fetch for photo {photo_id}")
        logging.debug(f"Flickr API initialized: {self.flickr is not None}")
        logging.debug(f"API Key present: {bool(os.environ.get('FLICKR_API_KEY'))}")

        if not self.flickr:
            logging.debug("No Flickr API connection, falling back to JSON")
            return self._load_json_metadata(photo_id)

        try:
            # Load base metadata from JSON
            metadata = self._load_json_metadata(photo_id) or {}

            # Get photo info from API
            photo_info = self.flickr.photos.getInfo(
                api_key=os.environ['FLICKR_API_KEY'],
                photo_id=photo_id
            )

            # Extract all API fields using dedicated methods
            self._extract_privacy_from_api(photo_info, metadata)
            self._extract_albums_from_api(photo_id, metadata)
            #self._extract_orientation_from_api(photo_info, metadata)

            return metadata

        except Exception as e:
            logging.debug(f"API fetch failed for photo {photo_id}, using JSON: {str(e)}")
            return self._load_json_metadata(photo_id)

    """
    def _extract_privacy_from_api(self, photo_info, metadata: Dict):
        #Extract privacy settings from API response using privacy level mapping
        visibility = photo_info.find('photo/visibility')
        if visibility is not None:
            privacy_level = 5  # Default to private (5)

            if int(visibility.get('ispublic', 0)):
                privacy_level = 1
            elif int(visibility.get('isfriend', 0)) and int(visibility.get('isfamily', 0)):
                privacy_level = 4
            elif int(visibility.get('isfriend', 0)):
                privacy_level = 2
            elif int(visibility.get('isfamily', 0)):
                privacy_level = 3

            # Map privacy level to string
            privacy_map = {
                1: 'public',
                2: 'friends only',
                3: 'family only',
                4: 'friends & family',
                5: 'private'
            }
            metadata['privacy'] = privacy_map[privacy_level]
            metadata['privacy_level'] = privacy_level  # Store numeric level for reference

    def _extract_albums_from_api(self, photo_id: str, metadata: Dict):
        #Extract album information using photosets.getPhotos
        try:
            albums = []
            # First get all photosets (albums) for the user
            photosets = self.flickr.photosets.getList(
                api_key=os.environ['FLICKR_API_KEY'],
                user_id=self.account_data.get('nsid')
            )

            # For each photoset, check if our photo is in it
            for photoset in photosets.findall('.//photoset'):
                photoset_id = photoset.get('id')
                photos = self.flickr.photosets.getPhotos(
                    api_key=os.environ['FLICKR_API_KEY'],
                    photoset_id=photoset_id,
                    user_id=self.account_data.get('nsid')
                )

                # Check if our photo_id is in this photoset
                for photo in photos.findall('.//photo'):
                    if photo.get('id') == photo_id:
                        albums.append(photoset.find('title').text)
                        break  # Found in this album, no need to check rest of photos

            if albums:
                self.photo_to_albums[photo_id] = albums
                metadata['albums'] = albums

        except Exception as e:
            logging.debug(f"Failed to get albums for {photo_id}: {str(e)}")
    """

    def _load_json_metadata(self, photo_id: str) -> Optional[Dict]:
        """Load metadata from JSON file only"""
        try:
            possible_patterns = [
                f"photo_{photo_id}.json",
                f"{photo_id}.json",
                f"{int(photo_id):d}.json"
            ]

            for pattern in possible_patterns:
                photo_file = self.metadata_dir / pattern
                if photo_file.exists():
                    with open(photo_file, 'r', encoding='utf-8') as f:
                        return json.load(f)

            return None

        except Exception as e:
            logging.error(f"Error loading JSON metadata for photo {photo_id}: {str(e)}")
            return None

     # Replace original _load_photo_metadata with API-first version
    _load_photo_metadata = _get_metadata_from_api


    # END OF API METADATA

    def _find_unorganized_photos(self) -> Dict[str, Path]:
        """
        Find all photos in the photos directory that aren't in any album
        Returns a dict mapping photo IDs to their file paths
        """
        try:
            #logging.critical("\nTESTING LOGGING LEVELS")  # This should always show
            #logging.error("This is an error test")       # This should show
            #logging.warning("This is a warning test")    # This should show
            #logging.info("This is an info test")         # This might show
            #logging.debug("This is a debug test")        # This might show

            # Get all media files in photos directory
            all_photos = {}  # photo_id -> Path
            unidentified_photos = []  # Files where we couldn't extract an ID

            logging.debug("\n=== Starting Unorganized Photos Analysis ===")

            # List all files in directory
            all_files = list(self.photos_dir.iterdir())
            logging.debug(f"Found {len(all_files)} total files in photos directory")

            # First, log what's in photo_to_albums
            logging.debug("\nCurrently organized photos:")
            logging.debug(f"photo_to_albums contains {len(self.photo_to_albums)} entries")
            for photo_id in list(self.photo_to_albums.keys())[:5]:  # Show first 5
                logging.debug(f"- Photo ID {photo_id} is in albums: {self.photo_to_albums[photo_id]}")

            # Get all media files in photos directory
            all_photos = {}  # photo_id -> Path
            unidentified_photos = []  # Files where we couldn't extract an ID

            logging.info("Scanning photos directory for unorganized photos...")

            # Process each file
            logging.debug("\nProcessing files in photos directory:")
            for file in all_files:
                if not file.is_file():
                    logging.debug(f"Skipping non-file: {file.name}")
                    continue

                if not self._is_supported_extension(file.name):
                    logging.debug(f"Skipping unsupported extension: {file.name}")
                    continue

                photo_id = self._extract_photo_id(file.name)
                if photo_id:
                    logging.debug(f"Found photo ID {photo_id} for file: {file.name}")
                    all_photos[photo_id] = file
                else:
                    logging.debug(f"Could not extract photo ID from: {file.name}")
                    unidentified_photos.append(file)

            # For files without IDs, generate sequential IDs
            for idx, file in enumerate(unidentified_photos):
                generated_id = f"unknown_{idx+1}"
                all_photos[generated_id] = file
                logging.info(f"Generated ID {generated_id} for file: {file.name}")

            # Find photos not in any album
            organized_photos = set(self.photo_to_albums.keys())
            unorganized_photos = {}

            logging.debug("\nComparing with organized photos:")
            logging.debug(f"Total photos found: {len(all_photos)}")
            logging.debug(f"Photos in albums: {len(organized_photos)}")

            for photo_id, file_path in all_photos.items():
                if photo_id not in organized_photos:
                    unorganized_photos[photo_id] = file_path
                    logging.debug(f"Found unorganized photo: {file_path.name} (ID: {photo_id})")

            # Enhanced logging
            if unorganized_photos:
                logging.info(f"Found {len(unorganized_photos)} photos not in any album")
                logging.info("Sample of unorganized photos:")
                for photo_id, file_path in list(unorganized_photos.items())[:5]:
                    logging.info(f"  - {photo_id}: {file_path.name}")

                # Log some statistics
                logging.info(f"\nPhoto Organization Statistics:")
                logging.info(f"Total files processed: {len(all_photos)}")
                logging.info(f"Files with extracted IDs: {len(all_photos) - len(unidentified_photos)}")
                logging.info(f"Files needing generated IDs: {len(unidentified_photos)}")
                logging.info(f"Organized photos: {len(organized_photos)}")
                logging.info(f"Unorganized photos: {len(unorganized_photos)}")
            else:
                logging.info("All photos are organized in albums")

            logging.debug("=== End Unorganized Photos Analysis ===\n")
            return unorganized_photos

        except Exception as e:
            logging.error(f"Error finding unorganized photos: {str(e)}")
            logging.exception("Full traceback:")
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
                # Create the 00_NoAlbum first
                no_album = {
                    'title': '00_NoAlbum',
                    'description': 'Photos not organized in any Flickr album',
                    'photos': []
                }
                self.albums.append(no_album)

                # Add each unorganized photo to the mapping and album
                for photo_id, file_path in unorganized_photos.items():
                    self.photo_to_albums[photo_id] = ['00_NoAlbum']
                    no_album['photos'].append(photo_id)  # Add to the album's photo list

                # Store the file paths for unorganized photos for later use
                self.unorganized_photo_paths = unorganized_photos

                total_photos = len(self.photo_to_albums)
                organized_count = sum(1 for p in self.photo_to_albums.values() if '00_NoAlbum' not in p)
                unorganized_count = len(unorganized_photos)

                logging.info(f"Photo organization summary:")
                logging.info(f"- Total photos: {total_photos}")
                logging.info(f"- In albums: {organized_count}")
                logging.info(f"- Not in albums: {unorganized_count}")
                logging.info(f"- Added to 00_NoAlbum: {len(no_album['photos'])}")

            else:
                logging.info("All photos are organized in albums")

        except Exception as e:
            logging.error(f"Error building photo-album mapping: {str(e)}")
            logging.exception("Full traceback:")
            self.photo_to_albums = {}  # Initialize empty if there's an error
            raise

    def __init__(self,
                    metadata_dir: str,
                    photos_dir: str,
                    output_dir: str,
                    date_format: str = 'yyyy/yyyy-mm-dd',
                    api_key: Optional[str] = None,
                    log_file: Optional[str] = None,
                    results_dir: Optional[str] = None,
                    include_extended_description: bool = INCLUDE_EXTENDED_DESCRIPTION,
                    write_xmp_sidecars: bool = WRITE_XMP_SIDECARS,
                    block_if_failure: bool = False,
                    resume: bool = False,
                    use_api: bool = False,
                    quiet: bool = False,
                    fave_weight: float = 10.0,
                    comment_weight: float = 3.0,
                    view_weight: float = 5.0,
                    min_views: int = 10,
                    min_faves: int = 2,
                    min_comments: int = 0,
                    cpu_cores: int = 4,
                    chunk_size: int = 100,
                    batch_size: int = 1000,
                    max_memory_percent: int = 75):

            # First, set all parameters as instance attributes
            self.block_if_failure = block_if_failure
            self.resume = resume
            self.date_format = date_format  # Now this will always have a value
            self.quiet = quiet
            self.use_api = use_api
            total_memory = psutil.virtual_memory().total
            self.max_memory = (total_memory * max_memory_percent) // 100
            self.batch_size = batch_size
            self.cpu_cores = cpu_cores

            logging.info(f"Maximum memory set to {self.max_memory / (1024*1024*1024):.1f}GB "
                        f"({max_memory_percent}% of system memory)")
            logging.info(f"Batch size set to {batch_size} photos")
            logging.info(f"Using {cpu_cores} CPU cores")

            # Track processing statistics
            self.stats = {
                'total_files': 0,
                'successful': {
                    'count': 0,
                    'details': set()  # Will store (source_file, dest_file, status) tuples
                },
                'failed': {
                    'count': 0,
                    'details': [],  # Top-level details for overall failures
                    'metadata': {
                        'count': 0,
                        'details': set()  # Will store (file, error_msg, exported) tuples
                    },
                    'file_copy': {
                        'count': 0,
                        'details': set()  # Will store (file, error_msg) tuples
                    }
                },
                'skipped': {
                    'count': 0,
                    'details': set()  # Will store (file, reason) tuples
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

            # Set debug logging level
            #logging.getLogger().setLevel(logging.DEBUG)
            #logging.debug("Debug logging enabled")

            # Configure a stream handler to show debug messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)

            # First, set all parameters as instance attributes
            self.block_if_failure = block_if_failure
            self.resume = resume  # Explicitly set the resume attribute
            self.date_format = date_format  # Also set date_format
            self.quiet = quiet

            self.include_extended_description = include_extended_description
            self.write_xmp_sidecars = write_xmp_sidecars

            # Setup logging with quiet parameter
            self._setup_logging(log_file, quiet)

            self.resume = resume

            # Initialize directories
            self.metadata_dir = Path(metadata_dir)
            self.photos_dir = Path(photos_dir)
            self.output_dir = Path(output_dir)
            self.results_dir = Path(results_dir) if results_dir else self.output_dir

            # Clear output directory if not resuming
            if not self.resume:
                self._clear_output_directory()

            # Add tracking sets
            self.processed_ids = set()  # Track unique photo IDs
            self.successful_ids = set()  # Track successfully processed IDs
            self.failed_ids = set()     # Track failed IDs

            # Create failed files directory
            self.failed_dir = self.output_dir / "failed_files"
            self.failed_dir.mkdir(parents=True, exist_ok=True)

            # Initialize data containers
            self.account_data = None
            self.user_mapping = {}
            self.photo_to_albums: Dict[str, List[str]] = {}

            #interestingness / hightlights
            self.fave_weight = fave_weight
            self.comment_weight = comment_weight
            self.view_weight = view_weight
            self.min_views = min_views
            self.min_faves = min_faves
            self.min_comments = min_comments

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
                    logging.warning("Comments and favorites lookup will be limited")
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

                # Just build photo ID map
                print("Building photo index...")
                #print(f"Photos directory: {photos_dir}")  # Add this line
                self.photo_id_map = {}
                photo_files = [f for f in self.photos_dir.iterdir() if f.is_file()]
                #print(f"Found {len(photo_files)} files in directory")  # Add this line
                for f in photo_files:
                    photo_id = self._extract_photo_id(f.name)
                    if photo_id:
                        self.photo_id_map[photo_id] = f
                    else:
                        print(f"Could not extract ID from: {f.name}")  # Add this line

                print(f"Successfully indexed {len(self.photo_id_map)} photos")  # Add this line

                # Initialize tracking sets
                self.unique_successful = set()
                self.unique_failed = set()

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
        """Configure logging with both file and console output"""

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL)  # Set to DEBUG to capture all levels

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set up format for logging
        console_format = '%(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Set up minimal console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        root_logger.addHandler(console_handler)

        # Only set up file logging if specifically requested
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)  # Only log errors to file
            root_logger.addHandler(file_handler)

        # Reduce logging from external libraries
        logging.getLogger('flickrapi').setLevel(logging.CRITICAL)
        logging.getLogger('requests').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('PIL').setLevel(logging.CRITICAL)

        # Log that debug logging is enabled
        logging.debug("Debug logging initialized")

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
                f.write(f"Total photos/videos processed: {self.stats['total_files']}\n")
                f.write(f"Successfully processed files: {self.stats['successful']['count']}\n")
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
        """Find JSON files matching various patterns"""
        # Move debug logging to logging.debug level
        logging.debug(f"Searching for {base_name} files")

        patterns = [
            f"{base_name}.json",
            f"{base_name}_part*.json",
            f"{base_name}*.json"
        ]

        matching_files = []
        for pattern in patterns:
            files = list(self.metadata_dir.glob(pattern))
            if files:
                logging.debug(f"Found {len(files)} files matching pattern {pattern}")
                for f in files:
                    logging.debug(f"  - {f.name}")
                matching_files.extend(files)

        unique_files = sorted(set(matching_files))

        if not unique_files:
            logging.debug(f"No {base_name} files found with any pattern")
        else:
            logging.debug(f"Found {len(unique_files)} unique {base_name} files")

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
        logging.info(f"Loading {base_name} files...")
        data_files = self._find_json_files(base_name)
        total_files = len(data_files)

        for i, data_file in enumerate(data_files, 1):
            logging.info(f"Processing file {i}/{total_files} ({(i/total_files)*100:.1f}%)")

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
        """Load the account profile data with progress indication"""
        with tqdm(total=100, desc="Loading account profile", unit="%") as pbar:
            profile_file = self.metadata_dir / 'account_profile.json'
            pbar.update(20)  # Show progress for finding file

            try:
                if not profile_file.exists():
                    logging.warning("account_profile.json not found, some metadata will be missing")
                    self.account_data = {}
                    pbar.update(80)  # Complete the progress bar
                    return

                pbar.update(30)  # Show progress for file check

                with open(profile_file, 'r', encoding='utf-8') as f:
                    self.account_data = json.load(f)
                    pbar.update(40)  # Show progress for loading

                    user_name = self.account_data.get('real_name', 'unknown user')
                    logging.info(f"Loaded account profile for {user_name}")

                    # Log some account details
                    logging.debug("Account details loaded:")
                    logging.debug(f"- Username: {self.account_data.get('screen_name', 'unknown')}")
                    logging.debug(f"- Join date: {self.account_data.get('join_date', 'unknown')}")
                    logging.debug(f"- NSID: {self.account_data.get('nsid', 'unknown')}")

                    pbar.update(10)  # Final progress update

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding account profile JSON: {str(e)}")
                self.account_data = {}
                pbar.update(100)  # Complete the progress bar

            except Exception as e:
                logging.error(f"Error loading account profile: {str(e)}")
                self.account_data = {}
                pbar.update(100)  # Complete the progress bar

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
        """Process user's photos and sort by engagement metrics"""
        try:
            print("\nAnalyzing photos for engagement metrics...")

            # Use pre-loaded metadata cache instead of re-reading files
            interesting_photos = []

            # Create temporary file for storing results
            temp_results_file = self.output_dir / "temp_highlights.json"

            # If resuming and temp file exists, load previous results
            if self.resume and temp_results_file.exists():
                try:
                    with open(temp_results_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass  # If loading fails, continue with normal processing

            total_files = len(self.photo_id_map)
            processed = 0
            meeting_criteria = 0

            print(f"Processing {total_files} photos")
            print("Parameters:")
            print(f"- Views: min {self.min_views} (weight: {self.view_weight})")
            print(f"- Favorites: min {self.min_faves} (weight: {self.fave_weight})")
            print(f"- Comments: min {self.min_comments} (weight: {self.comment_weight})")

            # Process metadata from cache
            for photo_id in self.photo_id_map:
                processed += 1

                # Load metadata for each photo
                photo_metadata = self._load_json_metadata(photo_id)
                if not photo_metadata:
                    continue

                if processed % 1000 == 0:
                    print(f"\rAnalyzing: {processed}/{total_files} ({(processed/total_files)*100:.1f}%) - "
                        f"Found {meeting_criteria} matching photos", end='')
                    sys.stdout.flush()

                try:
                    # Get metrics with default values
                    faves = int(photo_metadata.get('count_faves', 0))
                    comments = int(photo_metadata.get('count_comments', 0))
                    views = int(photo_metadata.get('count_views', 0))

                    # Check if meets criteria
                    if (views >= self.min_views or
                        faves >= self.min_faves or
                        comments >= self.min_comments):

                        # Calculate score
                        interestingness_score = (
                            (faves * self.fave_weight) +
                            (comments * self.comment_weight) +
                            (views * self.view_weight)
                        )

                        # Get source file from pre-loaded map
                        source_file = self.photo_id_map.get(photo_id)
                        if not source_file:
                            continue

                        photo_data = {
                            'id': photo_id,
                            'title': photo_metadata.get('name', ''),
                            'description': photo_metadata.get('description', ''),
                            'date_taken': photo_metadata.get('date_taken', ''),
                            'license': photo_metadata.get('license', ''),
                            'fave_count': faves,
                            'comment_count': comments,
                            'count_views': views,
                            'interestingness_score': interestingness_score,
                            'original_file': str(source_file),
                            'original': str(source_file),
                            'privacy': photo_metadata.get('privacy', ''),
                            'safety': photo_metadata.get('safety', '')
                        }
                        interesting_photos.append(photo_data)
                        meeting_criteria += 1

                except Exception as e:
                    continue

            print(f"\n\nFound {meeting_criteria} photos meeting criteria")

            if interesting_photos:
                # Sort by score
                interesting_photos.sort(key=lambda x: x['interestingness_score'], reverse=True)
                interesting_photos = interesting_photos[:per_page]

                # Save results to temp file
                try:
                    with open(temp_results_file, 'w') as f:
                        json.dump(interesting_photos, f)
                except Exception:
                    pass  # Continue even if saving fails

                return interesting_photos

            return []

        except Exception as e:
            print(f"Error analyzing photos: {str(e)}")
            return []

        finally:
            # Cleanup
            if 'temp_results_file' in locals() and temp_results_file.exists():
                try:
                    temp_results_file.unlink()
                except Exception:
                    pass

    def create_interesting_albums(self, time_period: str, photo_count: int = 100):
        """Create albums of user's most engaging photos"""
        try:
            # Create highlights_only parent directory
            highlights_dir = self.output_dir / "highlights_only"
            print(f"\nCreating highlights directory: {highlights_dir}")
            sys.stdout.flush()

            highlights_dir.mkdir(parents=True, exist_ok=True)

            # Get photos with engagement metrics
            print("\nAnalyzing photos for engagement metrics...")
            sys.stdout.flush()

            all_photos = self._fetch_user_interesting_photos(time_period, photo_count)
            if not all_photos:
                print("No photos found meeting engagement criteria")
                return

            print(f"\nFound {len(all_photos)} photos to process")
            print("Creating highlight albums...")
            sys.stdout.flush()

            # Initialize privacy groups
            privacy_groups = {
                'private': [],
                'friends & family': [],
                'friends only': [],
                'family only': [],
                'public': []
            }

            # Process each photo
            for photo in all_photos:
                # Get privacy value and normalize it
                raw_privacy = photo.get('privacy', '').lower().strip()

                # Map privacy values
                privacy_mapping = {
                    'private': 'private',
                    'friend & family': 'friends & family',
                    'friends & family': 'friends & family',
                    'friends&family': 'friends & family',
                    'friendandfamily': 'friends & family',
                    'friend and family': 'friends & family',
                    'friends and family': 'friends & family',
                    'friends': 'friends only',
                    'friend': 'friends only',
                    'family': 'family only',
                    'public': 'public',
                    '': 'private'
                }

                normalized_privacy = privacy_mapping.get(raw_privacy, 'private')
                privacy_groups[normalized_privacy].append(photo)

            # Create albums for each privacy group
            total_exported = 0
            for privacy_type, photos in privacy_groups.items():
                if photos:
                    print(f"\nProcessing {len(photos)} {privacy_type} photos...")
                    sys.stdout.flush()

                    folder_name = f"0{list(privacy_groups.keys()).index(privacy_type) + 1}_{privacy_type.replace(' ', '_').title()}_Highlights"

                    try:
                        self._create_single_interesting_album(
                            highlights_dir,
                            folder_name,
                            f"Your most engaging {privacy_type} Flickr photos",
                            photos
                        )
                        total_exported += len(photos)
                    except Exception as e:
                        print(f"Error creating album {folder_name}: {str(e)}")
                        continue

            print(f"\nHighlight albums creation complete!")
            print(f"Total photos exported: {total_exported}")
            sys.stdout.flush()

        except Exception as e:
            error_msg = f"Error creating highlight albums: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            logging.error(error_msg)
            raise

    def _create_single_interesting_album(self, highlights_dir: Path, folder_name: str, description: str, photos: List[Dict]):
        """Create a single album of engaging photos"""
        try:
            # Create folder for this privacy level
            album_dir = highlights_dir / folder_name
            album_dir.mkdir(parents=True, exist_ok=True)

            total_photos = len(photos)
            print(f"Processing {total_photos} photos for {folder_name}")
            sys.stdout.flush()

            # Update total files stat
            self.stats['total_files'] += total_photos

            processed = 0
            for i, photo in enumerate(photos, 1):
                try:
                    source_file = photo['original_file']
                    if not isinstance(source_file, Path):
                        source_file = Path(source_file)

                    if not source_file.exists():
                        self.stats['skipped']['count'] += 1
                        self.stats['skipped']['details'].append(
                            (str(source_file), "Source file not found")
                        )
                        continue

                    # Create filename without Flickr ID
                    if photo['title']:
                        safe_title = self._sanitize_folder_name(photo['title'])
                        photo_filename = f"{safe_title}{source_file.suffix}"
                    else:
                        photo_filename = self._get_destination_filename(photo['id'], source_file, photo)

                    dest_file = album_dir / photo_filename

                    # Handle filename conflicts
                    counter = 1
                    base_name = dest_file.stem
                    extension = dest_file.suffix
                    while dest_file.exists():
                        dest_file = album_dir / f"{base_name}_{counter}{extension}"
                        counter += 1

                    if i % 5 == 0 or i == total_photos:
                        print(f"\r{folder_name}: {i}/{total_photos} ({(i/total_photos)*100:.1f}%)", end='')
                        sys.stdout.flush()

                    # Copy file
                    shutil.copy2(source_file, dest_file)

                    # Prepare metadata
                    photo_metadata = photo.copy()
                    photo_metadata['original_file'] = str(source_file)
                    photo_metadata['original'] = str(source_file)

                    # Add engagement metrics to metadata
                    photo_metadata['engagement'] = {
                        'rank': i,
                        'total_ranked': total_photos,
                        'favorites': photo['fave_count'],
                        'comments': photo['comment_count'],
                        'views': photo.get('count_views', 0)
                    }

                    # Ensure photopage exists
                    if 'photopage' not in photo_metadata:
                        photo_metadata['photopage'] = f"https://www.flickr.com/photos/{self.account_data.get('nsid', '')}/{photo['id']}"

                    # Embed metadata based on media type
                    media_type = self.get_media_type(dest_file)
                    if media_type == MediaType.IMAGE:
                        self._embed_image_metadata(dest_file, photo_metadata)
                    elif media_type == MediaType.VIDEO:
                        self._embed_video_metadata(dest_file, photo_metadata)

                    if self.write_xmp_sidecars:
                        self._write_xmp_sidecar(dest_file, photo_metadata)

                    processed += 1
                    self.stats['successful']['count'] += 1
                    self.stats['successful']['details'].append(
                        (str(source_file), str(dest_file), "Highlight photo processed successfully")
                    )

                except Exception as e:
                    error_msg = f"Error processing highlight photo {photo.get('id', 'unknown')}: {str(e)}"
                    self.stats['failed']['count'] += 1
                    self.stats['failed']['details'].append(
                        (str(source_file) if 'source_file' in locals() else "unknown", error_msg)
                    )
                    logging.error(error_msg)
                    continue

            print(f"\nCompleted {folder_name}: {processed}/{total_photos} photos processed successfully")
            sys.stdout.flush()

        except Exception as e:
            error_msg = f"Error creating album {folder_name}: {str(e)}"
            print(error_msg)
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

    def _find_metadata_file(self, photo_id: str) -> Optional[Path]:
        """Helper method to find metadata file for a given ID"""
        patterns = [
            f"photo_{photo_id}.json",
            f"{photo_id}.json"
        ]

        for pattern in patterns:
            metadata_file = self.metadata_dir / pattern
            if metadata_file.exists():
                return metadata_file
        return None

    def _extract_photo_id(self, filename: str) -> Optional[str]:
        """
        Extract photo ID from filename and verify against metadata existence.
        Returns the ID that has matching metadata.
        """
        try:
            #print(f"Attempting to extract ID from: {filename}")  # Add this line
            # Convert filename to lowercase for consistent matching
            filename = filename.lower()

            # Find ALL potential IDs in the filename, including those not following _o pattern
            import re
            # Updated pattern to catch more ID variations
            patterns = [
                r'_(\d{10,11})(?:_o)?(?:\.|_)',  # Standard pattern
                r'[^0-9](\d{10,11})[^0-9]',      # Any 10-11 digit number
            ]

            all_matches = set()
            for pattern in patterns:
                matches = re.findall(pattern, filename)
                all_matches.update(matches)

            if not all_matches:
                return None

            # Convert matches to list and sort by length (to prioritize consistent length IDs)
            matches = sorted(all_matches, key=len, reverse=True)

            if len(matches) >= 1:
                # First, try to find metadata for any of the IDs
                for photo_id in matches:
                    if self._find_metadata_file(photo_id):
                        if len(matches) > 1:
                            logging.debug(f"Multiple IDs in {filename}, using {photo_id} (found metadata)")
                        return photo_id

                # If no metadata found but we have multiple IDs, use the largest (most recent) one
                if len(matches) > 1:
                    largest_id = str(max(int(id_) for id_ in matches))
                    logging.debug(f"No metadata found for any ID in {filename}, using largest: {largest_id}")
                    return largest_id

                # If single ID, return it even if no metadata (for consistent behavior)
                return matches[0]

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

            # Find unorganized photos and add to '00_NoAlbum'
            unorganized_photos = self._find_unorganized_photos()
            if unorganized_photos:
                # Create the 'No Album' album if unorganized photos exist
                no_album_album = {
                    'title': '00_NoAlbum',
                    'description': 'Photos not organized in any Flickr album',
                    'photos': []
                }
                self.albums.append(no_album_album)

                # Add each unorganized photo to the mapping and album
                for photo_id, file_path in unorganized_photos.items():
                    if photo_id not in self.photo_to_albums:
                        self.photo_to_albums[photo_id] = ['00_NoAlbum']
                        no_album_album['photos'].append(photo_id)  # Add to the album's photo list

            logging.info(f"Photo-album mapping complete. Total unique media items: {len(self.photo_to_albums)}")
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


    def create_album_structure(self) -> Path:
        """Create the album folder structure and return working directory"""
        try:
            full_export_dir = self.output_dir / "full_library_export" / "by_album"
            print(f"Creating album structure in: {full_export_dir}")
            sys.stdout.flush()

            # Create the directory
            full_export_dir.mkdir(parents=True, exist_ok=True)

            # Verify directory exists
            if not full_export_dir.exists():
                raise ValueError(f"Failed to create directory: {full_export_dir}")

            # Create album directories under by_album
            if hasattr(self, 'albums'):
                for album in self.albums:
                    album_dir = full_export_dir / self._sanitize_folder_name(album['title'])
                    album_dir.mkdir(parents=True, exist_ok=True)

                    if not album_dir.exists():
                        raise ValueError(f"Failed to create album directory: {album_dir}")

                print(f"Created {len(self.albums)} album directories")
                sys.stdout.flush()

                return full_export_dir
            else:
                raise ValueError("No albums data found")

        except Exception as e:
            error_msg = f"Error creating album structure: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            raise

    def create_date_structure(self, date_format: str) -> Path:
        """Create the date-based folder structure and return working directory"""
        try:
            base_dir = self.output_dir / "full_library_export"
            working_dir = base_dir / "by_date"

            print(f"Creating date structure in: {working_dir}")
            sys.stdout.flush()

            # Create base directories
            for dir_path in [self.output_dir, base_dir, working_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    raise ValueError(f"Failed to create directory: {dir_path}")

            return working_dir

        except Exception as e:
            error_msg = f"Error creating date structure: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            raise

    def _get_date_path(self, date_str: str, date_format: str) -> Path:
        """Convert date string to folder path based on format"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

            if date_format == 'yyyy/yyyy-mm-dd':
                return Path(f"{date.year}/{date.year:04d}-{date.month:02d}-{date.day:02d}")
            elif date_format == 'yyyy/yyyy-mm':
                return Path(f"{date.year}/{date.year:04d}-{date.month:02d}")
            elif date_format == 'yyyy-mm-dd':
                return Path(f"{date.year:04d}-{date.month:02d}-{date.day:02d}")
            elif date_format == 'yyyy-mm':
                return Path(f"{date.year:04d}-{date.month:02d}")
            elif date_format == 'yyyy':
                return Path(f"{date.year:04d}")
            else:
                raise ValueError(f"Unsupported date format: {date_format}")

        except Exception as e:
            logging.error(f"Error processing date {date_str}: {str(e)}")
            return Path("unknown_date")

    def _sanitize_folder_name(self, name: str) -> str:
        """Convert filename/foldername to safe version"""
        # Replace spaces with underscores and remove/replace special characters
        sanitized = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                        for c in name.replace(' ', '_'))

        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        # Ensure name isn't empty
        if not sanitized:
            sanitized = "unnamed"
        return sanitized

    def process_photos(self, organization: str, date_format: str = None):
        """Process photos with enhanced debugging"""
        try:
            self.unique_total = len(set(self.photo_to_albums.keys()))
            self.start_time = time.time()

            print(f"\nProcessing {self.unique_total} photos")
            sys.stdout.flush()

            if organization == 'by_date':
                working_dir = self.create_date_structure(date_format)
            else:
                working_dir = self.create_album_structure()

            photo_items = list(self.photo_to_albums.items())
            total_processed = 0

            for batch_idx, batch_start in enumerate(range(0, len(photo_items), self.batch_size), 1):
                batch_end = min(batch_start + self.batch_size, len(photo_items))
                current_batch = photo_items[batch_start:batch_end]

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
                    futures = []
                    for photo_id, albums in current_batch:
                        if organization == 'by_date':
                            future = executor.submit(
                                self._process_single_photo_by_date,
                                photo_id, date_format, working_dir
                            )
                        else:
                            future = executor.submit(
                                self._process_single_photo,
                                (photo_id, albums, working_dir)
                            )
                        futures.append((future, photo_id))

                    # Process results for this batch
                    for future, photo_id in futures:
                        try:
                            result = future.result()
                            if result and isinstance(result, tuple):
                                source_file, dest_file = result
                                if dest_file and dest_file.exists():
                                    self.unique_successful.add(photo_id)
                                else:
                                    self.unique_failed.add(photo_id)
                            else:
                                self.unique_failed.add(photo_id)

                            total_processed += 1

                        except Exception as e:
                            self.unique_failed.add(photo_id)
                            total_processed += 1

                    # Update progress after processing batch
                    elapsed_time = time.time() - self.start_time
                    photos_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
                    remaining_photos = self.unique_total - total_processed
                    estimated_remaining = remaining_photos / photos_per_second if photos_per_second > 0 else 0

                    elapsed = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
                    remaining = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining))

                    # Show current batch info and overall progress
                    if organization == 'by_date':
                        current_location = f"Processing date structure - Batch {batch_idx}"
                    else:
                        current_location = f"Processing albums - Batch {batch_idx}"

                    print(f"\r{current_location} | Completed {total_processed} of {self.unique_total}")
                    print(f"\rTotal: {total_processed}/{self.unique_total} files | "
                        f"Success: {len(self.unique_successful)} | "
                        f"Failed: {len(self.unique_failed)} | "
                        f"Time elapsed: {elapsed} | "
                        f"Estimated remaining: {remaining}", end='\r')
                    sys.stdout.flush()

                self._cleanup_memory()

            print("\n\nProcessing Complete")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error in photo processing: {str(e)}")
            raise

    def _cleanup_memory(self):
        """Perform memory cleanup between batches"""
        gc.collect()

        # Clear caches
        self.user_info_cache.clear()

        # Log memory usage
        #process = psutil.Process()
       # memory_info = process.memory_info()
       # logging.debug(f"Memory usage after cleanup: {memory_info.rss / (1024*1024):.1f}MB")

    def _check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds threshold"""
        try:
            memory_info = psutil.Process().memory_info()
            system_memory = psutil.virtual_memory()

            memory_usage = memory_info.rss
            memory_limit = self.max_memory

            logging.debug(f"Memory usage: {memory_usage / (1024*1024):.1f}MB")
            logging.debug(f"Memory limit: {memory_limit / (1024*1024):.1f}MB")

            return memory_usage > memory_limit

        except Exception as e:
            logging.warning(f"Error checking memory: {str(e)}")
            return False

    def _update_progress(self, processed: int, total: int):
        """Update progress with current statistics"""
        percentage = (processed / total) * 100
        progress_msg = (
            f"\rProgress: {processed}/{total} ({percentage:.1f}%) - "
            f"Success: {self.stats['successful']['count']}, "
            f"Failed: {self.stats['failed']['count']}"
        )
        print(progress_msg, end='', flush=True)
        logging.info(progress_msg.replace('\r', ''))

    def _save_checkpoint(self, processed: int, batch_end: int):
        """Save processing checkpoint"""
        try:
            checkpoint = {
                'processed': processed,
                'batch_end': batch_end,
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }

            checkpoint_file = self.results_dir / 'processing_checkpoint.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            logging.debug(f"Saved checkpoint at {processed} photos")

        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {str(e)}")

    def _print_final_stats(self, processed: int, total: int):
        """Print final processing statistics"""
        print("\nProcessing complete!")
        print(f"Processed {processed}/{total} photos")
        print(f"Successful: {self.stats['successful']['count']}")
        print(f"Failed: {self.stats['failed']['count']}")

        memory_info = psutil.Process().memory_info()
        logging.info(f"Final memory usage: {memory_info.rss / (1024*1024):.1f}MB")

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


    def _handle_failed_file(self, photo_id: str, source_file: Optional[Path], error_reason: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Handle a failed file by copying it to the failed directory and logging the error"""
        try:
            if source_file and source_file.exists():
                # Create failed_files directory
                self.failed_dir.mkdir(parents=True, exist_ok=True)

                # Create destination path using same date structure as main export
                dest_file = self.failed_dir / source_file.name

                # Handle filename conflicts
                counter = 1
                while dest_file.exists():
                    dest_file = self.failed_dir / f"{source_file.stem}_{counter}{source_file.suffix}"
                    counter += 1

                # Copy the file
                try:
                    shutil.copy2(source_file, dest_file)

                    # Create an error info file
                    error_info_file = dest_file.with_suffix('.error.txt')
                    with open(error_info_file, 'w', encoding='utf-8') as f:
                        f.write(f"Photo ID: {photo_id}\n")
                        f.write(f"Original path: {source_file}\n")
                        f.write(f"Error: {error_reason}\n")
                        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                    return source_file, dest_file

                except Exception as copy_error:
                    print(f"Error copying failed file: {str(copy_error)}")
                    return None, None

        except Exception as e:
            print(f"Error handling failed file {photo_id}: {str(e)}")

        return None, None

    #by album processing method
    def _process_single_photo(self, args: Tuple) -> Tuple[Optional[Path], Optional[Path]]:
        """Process a single photo and copy to all its album locations"""
        try:
            photo_id, albums, working_dir = args

            if not working_dir:
                return self._handle_failed_file(photo_id, None, "No working directory")

            # Use pre-loaded maps
            source_file = self.photo_id_map.get(photo_id)
            if not source_file:
                return self._handle_failed_file(photo_id, None, "Source file not found")

            photo_json = self._load_json_metadata(photo_id)
            if not photo_json:
                return self._handle_failed_file(photo_id, source_file, "No metadata found")

            # Process once for all albums
            processed_files = []
            for album_name in albums:
                try:
                    # Determine album directory
                    if album_name == '00_NoAlbum':
                        date_taken = photo_json.get('date_taken')
                        if not date_taken:
                            date_taken = datetime.fromtimestamp(source_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        date_path = self._get_date_path(date_taken, self.date_format)
                        album_dir = working_dir / "00_NoAlbum" / date_path
                    else:
                        album_dir = working_dir / self._sanitize_folder_name(album_name)

                    album_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = album_dir / self._get_destination_filename(photo_id, source_file, photo_json)

                    # Copy file
                    shutil.copy2(source_file, dest_file)

                    # Only embed metadata for the first copy
                    if not processed_files:
                        media_type = self.get_media_type(dest_file)
                        if media_type == MediaType.IMAGE:
                            self._embed_image_metadata(dest_file, photo_json)
                            if self.write_xmp_sidecars:
                                self._write_xmp_sidecar(dest_file, photo_json)
                        elif media_type == MediaType.VIDEO:
                            self._embed_video_metadata(dest_file, photo_json)
                            if self.write_xmp_sidecars:
                                self._write_xmp_sidecar(dest_file, photo_json)

                    processed_files.append(dest_file)

                except Exception as e:
                    return self._handle_failed_file(photo_id, source_file, f"Album processing error: {str(e)}")

            return source_file, processed_files[0] if processed_files else None

        except Exception as e:
            return self._handle_failed_file(photo_id, source_file if 'source_file' in locals() else None,
                                          f"Processing error: {str(e)}")

    #by date processing method
    def _process_single_photo_by_date(self, photo_id: str, date_format: str, working_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Process a single photo with date-based organization"""
        try:
            # Use pre-loaded maps
            source_file = self.photo_id_map.get(photo_id)
            if not source_file:
                return self._handle_failed_file(photo_id, None, "Source file not found")

            photo_json = self._load_json_metadata(photo_id)

            if not photo_json:
                return self._handle_failed_file(photo_id, source_file, "No metadata found")

            # Get date and create directory
            try:
                if photo_json and 'date_taken' in photo_json:
                    date_taken = photo_json['date_taken']
                else:
                    date_taken = datetime.fromtimestamp(source_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                date_taken = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            date_path = self._get_date_path(date_taken, date_format)
            date_dir = working_dir / date_path
            date_dir.mkdir(parents=True, exist_ok=True)

            dest_file = date_dir / self._get_destination_filename(photo_id, source_file, photo_json)

            # Process file
            shutil.copy2(source_file, dest_file)

            if photo_json:
                media_type = self.get_media_type(dest_file)
                if media_type == MediaType.IMAGE:
                    self._embed_image_metadata(dest_file, photo_json)
                    if self.write_xmp_sidecars:
                        self._write_xmp_sidecar(dest_file, photo_json)
                elif media_type == MediaType.VIDEO:
                    self._embed_video_metadata(dest_file, photo_json)
                    if self.write_xmp_sidecars:
                        self._write_xmp_sidecar(dest_file, photo_json)

            return source_file, dest_file

        except Exception as e:
            return self._handle_failed_file(photo_id, source_file if 'source_file' in locals() else None,
                                          f"Processing error: {str(e)}")

    def _sanitize_filename(self, filename: str) -> str:
        """Convert filename to filesystem-safe version"""
        try:
            # Remove or replace invalid characters
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')

            # Remove leading/trailing spaces and periods
            filename = filename.strip('. ')

            # Replace multiple spaces/underscores with single underscore
            filename = re.sub(r'[\s_]+', '_', filename)

            # Ensure filename isn't empty
            if not filename or filename.startswith('.'):
                return 'untitled_photo'

            # Limit filename length (max 255 chars is common filesystem limit)
            if len(filename) > 255:
                name, ext = os.path.splitext(filename)
                filename = name[:255 - len(ext)] + ext

            return filename

        except Exception as e:
            logging.error(f"Error sanitizing filename '{filename}': {str(e)}")
            return 'untitled_photo'

    def _embed_image_metadata(self, photo_file: Path, metadata: Dict):
        """Embed metadata into an image file using exiftool"""
        try:
            args = self._build_exiftool_args(photo_file, metadata)

            # First verify JPEG integrity if it's a JPEG file
            if photo_file.suffix.lower() in ['.jpg', '.jpeg']:
                is_valid, error_msg = JPEGVerifier.is_jpeg_valid(str(photo_file))
                if not is_valid:
                    if not JPEGVerifier.attempt_repair(str(photo_file)):
                        raise ValueError(f"Invalid JPEG file: {error_msg}")

            # Run exiftool
            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode != 0:
                raise ValueError(f"Exiftool error: {result.stderr}")

            if result.stderr and 'error' in result.stderr.lower():
                logging.warning(f"Exiftool warning for {photo_file}: {result.stderr}")

        except Exception as e:
            error_msg = f"Error embedding metadata in {photo_file}: {str(e)}"
            logging.error(error_msg)
            raise

    def _load_photo_metadata(self, photo_id: str) -> Optional[Dict]:
       """Load metadata for a specific photo"""
       metadata = self._load_json_metadata(photo_id)
       if not metadata:
           return None

       if self.use_api and self.flickr:
           try:
               photo_info = self.flickr.photos.getInfo(
                   api_key=os.environ['FLICKR_API_KEY'],
                   photo_id=photo_id
               )
               """
               self._extract_privacy_from_api(photo_info, metadata)
               self._extract_albums_from_api(photo_id, metadata)
               self._extract_orientation_from_api(photo_info, metadata)
               """

           except Exception as e:
               if not self.quiet:
                   logging.debug(f"API fetch failed for photo {photo_id}")

       return metadata

    def _load_json_metadata(self, photo_id: str) -> Optional[Dict]:
        """Load metadata for a photo, handling multiple possible IDs"""
        try:
            # First try direct metadata file
            metadata_file = self._find_metadata_file(photo_id)
            if metadata_file:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

            # If no direct match, look for files that might contain this as original ID
            all_metadata_files = list(self.metadata_dir.glob("photo_*.json"))
            for mfile in all_metadata_files:
                try:
                    with open(mfile, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check various fields where the ID might appear
                        if (str(data.get('id', '')) == photo_id or
                            str(data.get('original_id', '')) == photo_id or
                            photo_id in [str(x) for x in data.get('related_ids', [])] or
                            photo_id in str(data.get('original', ''))):
                            return data
                except:
                    continue

            logging.debug(f"No metadata found for photo {photo_id}")
            return None

        except Exception as e:
            logging.error(f"Error loading JSON metadata for photo {photo_id}: {str(e)}")
            return None

    def _find_photo_file(self, photo_id: str, filename: str) -> Optional[Path]:
        """Find the original photo file using the Flickr ID with enhanced logging"""
        logging.debug(f"Searching for photo file {photo_id} (filename: {filename})")
        logging.debug(f"Looking in directory: {self.photos_dir}")


        # Normalize photo_id and filename for case-insensitive comparison
        photo_id = photo_id.lower()  # Make photo ID lowercase
        normalized_filename = filename.lower()  # Make filename lowercase

        # First try: exact match with photo ID (case insensitive)
        matches = []
        for file in self.photos_dir.iterdir():
            if f"_{photo_id}_" in file.name.lower() or f"_{photo_id}." in file.name.lower():
                matches.append(file)
                logging.debug(f"Found exact match: {file.name}")

        if matches:
            if len(matches) > 1:
                logging.warning(f"Multiple matches found for {photo_id}, using first one: {matches[0]}")
            return matches[0]

        # Second try: normalize ID and try again (case insensitive)
        normalized_id = photo_id.lstrip('0')  # Remove leading zeros
        logging.debug(f"Trying normalized ID: {normalized_id}")

        for file in self.photos_dir.iterdir():
            file_parts = file.name.lower().split('_')  # Make lowercase for comparison
            for part in file_parts:
                clean_part = part.split('.')[0]  # Remove extension if present
                if clean_part.lstrip('0') == normalized_id:
                    logging.debug(f"Found match with normalized ID: {file.name}")
                    return file

        # Third try: look for filename (case insensitive)
        for file in self.photos_dir.iterdir():
            if file.name.lower().startswith(normalized_filename):
                logging.debug(f"Found match by filename: {file.name}")
                return file

        # If we get here, we couldn't find the file
        logging.debug("Listing sample of files in photos directory:")
        all_files = list(self.photos_dir.iterdir())
        logging.debug(f"Total files in directory: {len(all_files)}")
        if all_files:
            for f in all_files[:5]:
                logging.debug(f"  - {f.name} (lowercase: {f.name.lower()})")

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

            # Run exiftool
            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode != 0:
                raise ValueError(f"Exiftool error: {result.stderr}")

            if result.stderr and 'error' in result.stderr.lower():
                logging.warning(f"Exiftool warning for {video_file}: {result.stderr}")

        except Exception as e:
            error_msg = f"Error embedding metadata in {video_file}: {str(e)}"
            logging.error(error_msg)
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
            <flickr:accountInfo rdf:parseType="Resource">        return self._han
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

        if media_file.suffix.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.JPG', '.JPEG', '.TIFF', '.TIF']:
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

                        current_orientation = exif.get(orientation_tag)
                        width, height = img.size
                        is_portrait = height > width

                        logging.debug(f"Processing orientation for {media_file}")
                        logging.debug(f"Current EXIF orientation: {current_orientation}")
                        logging.debug(f"Image dimensions: {width}x{height} (Portrait: {is_portrait})")

                        # Determine orientation
                        if current_orientation is None:
                            new_orientation = 1  # Default if no EXIF orientation
                        elif 'rotation' in metadata:
                            rotation_degrees = int(metadata["rotation"])
                            logging.debug(f"Flickr rotation value: {rotation_degrees}")

                            if rotation_degrees > 0:
                                rotation_to_orientation = {
                                    90: 6,    # Rotate 90 CW
                                    180: 3,   # Rotate 180
                                    270: 8    # Rotate 270 CW
                                }
                                new_orientation = rotation_to_orientation.get(rotation_degrees, 1)
                            else:
                                new_orientation = current_orientation
                        else:
                            new_orientation = current_orientation

                        # Handle combined rotations
                        if current_orientation and new_orientation != current_orientation:
                            # Calculate combined rotation
                            current_rotation = EXIF_ORIENTATION_MAP.get(current_orientation, {}).get('rotation', 0)
                            new_rotation = EXIF_ORIENTATION_MAP.get(new_orientation, {}).get('rotation', 0)
                            total_rotation = (current_rotation + new_rotation) % 360

                            # Map total rotation back to orientation
                            rotation_to_final = {
                                0: 1,
                                90: 6,
                                180: 3,
                                270: 8
                            }
                            new_orientation = rotation_to_final.get(total_rotation, 1)

                        # Log the orientation change
                        if current_orientation != new_orientation:
                            old_desc = EXIF_ORIENTATION_MAP.get(current_orientation, {}).get('description', 'Unknown')
                            new_desc = EXIF_ORIENTATION_MAP.get(new_orientation, {}).get('description', 'Unknown')
                            logging.debug(f"Changing orientation from {current_orientation} ({old_desc}) "
                                        f"to {new_orientation} ({new_desc})")

                        # Add orientation commands to exiftool args
                        args.extend([
                            f'-IFD0:Orientation#={new_orientation}',
                            '-IFD0:YCbCrPositioning=1',
                            '-IFD0:YCbCrSubSampling=2 2'
                        ])

                        # If the image is mirrored in its target orientation, add mirroring command
                        if EXIF_ORIENTATION_MAP.get(new_orientation, {}).get('mirrored', False):
                            args.extend(['-Flop'])  # Flop is horizontal mirroring

            except Exception as e:
                logging.warning(f"Error handling image orientation for {media_file}: {str(e)}")
                logging.debug(f"Exception details:", exc_info=True)

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
        # Create a console handler that will show these stats regardless of quiet flag
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)

        # Store the original handlers
        original_handlers = logging.getLogger().handlers
        logging.getLogger().handlers = [console_handler]

        try:
            # Print main statistics
            logging.info("\nProcessing Statistics:")
            logging.info("-" * 20)
            logging.info(f"Total unique photos: {self.unique_total}")
            logging.info(f"Successfully processed: {len(self.unique_successful)}")
            logging.info(f"Failed: {len(self.unique_failed)}")
            logging.info(f"Skipped: {self.stats['skipped']['count']}")
            logging.info("-" * 20)

            # Show failure details if there are any failures
            if len(self.unique_failed) > 0:
                # Restore original handlers for error reporting
                logging.getLogger().handlers = original_handlers

                logging.error("\nFailure Details:")
                # Show metadata failures
                if self.stats['failed']['metadata']['details']:
                    logging.error("Metadata failures:")
                    for file, error, exported in self.stats['failed']['metadata']['details'][:3]:
                        logging.error(f"- {file}: {error} (File exported: {'Yes' if exported else 'No'})")
                    if len(self.stats['failed']['metadata']['details']) > 3:
                        logging.error(f"... and {len(self.stats['failed']['metadata']['details']) - 3} more metadata failures")

                # Show file copy failures
                if self.stats['failed']['file_copy']['details']:
                    logging.error("File copy failures:")
                    for file, error in self.stats['failed']['file_copy']['details'][:3]:
                        logging.error(f"- {file}: {error}")
                    if len(self.stats['failed']['file_copy']['details']) > 3:
                        logging.error(f"... and {len(self.stats['failed']['file_copy']['details']) - 3} more file copy failures")

                logging.error("\nSee processing_results.txt for complete details")

        finally:
            # Restore original handlers
            logging.getLogger().handlers = original_handlers

def gui_main():
    """Entry point for GUI version"""
    try:
        if '--cli' in sys.argv:
            logging.info("Running in CLI mode")
            sys.argv.remove('--cli')
            # Call the main function directly without Gooey/wx
            main()
            return

        # Create wx.App instance before using wx
        app = wx.App(False)

        # Run with Gooey
        Gooey(
            program_name="Flickr to Any Tool",
            default_size=(1000, 800),
            minimum_size=(800, 600),
            resizable=True,
            navigation='TABBED',
            sidebar_title='Steps',
            terminal_font_family='Courier New',
            terminal_font_size=8,
            show_restart_button=False,
            tabbed_groups=True,
            progress_regex=r"(\d+)%",
            progress_expr="min(100, int(float({\1})))",
            hide_progress_msg=False,
            console_height=200,
            show_failure_modal=True,
            group_by_type=True,
            scrollable=True,
            fullscreen=False,
            header_height=80,
            header_show_title=True,
            header_bg_color='#3c3f41',
            terminal_panel_color='#F0F0F0',
            show_success_modal=False
        )(main)()

    except Exception as e:
        error_msg = f"Error in gui_main: {str(e)}\n"
        error_msg += "Traceback:\n"
        import traceback
        error_msg += traceback.format_exc()
        print(error_msg)
        logging.error(error_msg)
        raise

def main():
    import sys
    import argparse
    from gooey import GooeyParser

    parser = GooeyParser(description='Flickr to Any Tool')

    # Create groups
    preprocessing = parser.add_argument_group(
        'Step 1: Preprocessing',
        'Extract Flickr export zip files',
        gooey_options={'show_border': True}
    )

    # Add other non-directory arguments to preprocessing
    preprocessing.add_argument(
        '--zip-preprocessing',
        metavar='Enable ZIP Preprocessing',
        action='store_true',
        help='Enable if you need to extract Flickr export zip files first',
        default=True,
        gooey_options={'checkbox_label': 'Process ZIP files'}
    )

    main_settings = parser.add_argument_group(
        'Step 2: Main Settings',
        'Configure main conversion options',
        gooey_options={'show_border': True}
    )
    # Setup directory widgets
    preprocessing, main_settings = setup_directory_widgets(preprocessing, main_settings)

    # Create export type group
    export_type = parser.add_argument_group(
        'Step 3: Export Type',
        'Choose what to export',
        gooey_options={'show_border': True}
    )

    # Main settings non-directory arguments
    export_type.add_argument(
        '--organization',
        metavar='Organization Method',
        choices=['by_album', 'by_date'],
        default='by_date',
        help='How to organize photos in the library'
    )
    export_type.add_argument(
        '--date-format',
        metavar='Date Format',
        choices=['yyyy', 'yyyy-mm', 'yyyy/yyyy-mm-dd', 'yyyy/yyyy-mm', 'yyyy-mm-dd'],
        default='yyyy/yyyy-mm',
        help='Date format for folder structure'
    )

    export_type.add_argument(
        '--export-mode',
        metavar='What to Export',
        choices=[
            'Full library and Highlights',
            'Full library only',
            'Highlights only'
        ],
        default='Full library and Highlights',
        help='Choose what to export from your Flickr library',
        gooey_options={
            'label': 'Export Mode'
        }
    )

    # Interesting photos configuration
    export_type.add_argument(
        '--interesting-period',
        metavar='Interesting Time Period',
        choices=['all-time', 'byyear'],
        default='all-time',
        help='Time period for interesting photos'
    )
    export_type.add_argument(
        '--interesting-count',
        metavar='Number of Photos',
        type=int,
        default=100,
        help='Number of interesting photos to fetch (max 500)'
    )

    # Add highlight weighting settings
    export_type.add_argument(
        '--fave-weight',
        metavar='Favorite Weight',
        type=float,
        default=2,
        help='Weight multiplier for favorites (default: 10.0)',
        gooey_options={
            'label': 'Favorite Weight'
        }
    )

    export_type.add_argument(
        '--comment-weight',
        metavar='Comment Weight',
        type=float,
        default=1,
        help='Weight multiplier for comments (default: 5.0)',
        gooey_options={
            'label': 'Comment Weight'
        }
    )

    export_type.add_argument(
        '--view-weight',
        metavar='View Weight',
        type=float,
        default=2,
        help='Weight multiplier for views (default: 0.1)',
        gooey_options={
            'label': 'View Weight'
        }
    )

    # Add minimum threshold settings
    export_type.add_argument(
        '--min-views',
        metavar='Minimum Views',
        type=int,
        default=20,
        help='Minimum views required (default: 20)',
        gooey_options={
            'label': 'Minimum Views'
        }
    )

    export_type.add_argument(
        '--min-faves',
        metavar='Minimum Favorites',
        type=int,
        default=1,
        help='Minimum favorites required (default: 0)',
        gooey_options={
            'label': 'Minimum Favorites'
        }
    )

    export_type.add_argument(
        '--min-comments',
        metavar='Minimum Comments',
        type=int,
        default=1,
        help='Minimum comments required (default: 0)',
        gooey_options={
            'label': 'Minimum Comments'
        }
    )

    # Advanced options
    advanced = parser.add_argument_group(
        'Step 4: Advanced Options',
        'Configure additional settings',
        gooey_options={'show_border': True}
    )

    advanced.add_argument(
        '--no-extended-description',
        metavar='Skip Extended Description',
        action='store_true',
        help='Only include original description',
        gooey_options={'checkbox_label': 'Skip extended description'}
    )
    advanced.add_argument(
        '--no-xmp-sidecars',
        metavar='Skip XMP Files',
        action='store_true',
        help='Skip writing XMP sidecar files',
        gooey_options={'checkbox_label': 'Skip XMP sidecar files'}
    )
    advanced.add_argument(
        '--export-block-if-failure',
        metavar='Block on Failure',
        action='store_true',
        help='Stop if metadata processing fails',
        gooey_options={'checkbox_label': 'Stop on metadata failure'}
    )
    advanced.add_argument(
        '--resume',
        metavar='Resume Previous',
        action='store_true',
        help='Skip existing files',
        gooey_options={'checkbox_label': 'Resume previous run'}
    )
    advanced.add_argument(
        '--quiet',
        metavar='Quiet Mode',
        action='store_true',
        default=True,
        help='Reduce console output',
        gooey_options={'checkbox_label': 'Quiet mode'}
    )
    advanced.add_argument(
        '--use-api',
        metavar='Use Flickr API for comments and favorites',
        action='store_true',
        default=False,
        help='Use Flickr API for looking up comments and favorites',
        gooey_options={
            'checkbox_label': 'Use Flickr API'
        }
    )
    advanced.add_argument(
        '--api-key',
        metavar='Flickr API Key',
        help='Enter your Flickr API key',
        default='',
        gooey_options={
            'visible_when': {
                'use-api': True  # Only show when use-api is checked
            }
        }
    )

    advanced.add_argument(
        '--cpu-cores',
        metavar='CPU Cores',
        type=int,
        default=min(os.cpu_count(), 4),  # Default to lesser of 4 or available cores
        help=f'Number of CPU cores to use (default: {min(os.cpu_count(), 4)}, max: {os.cpu_count()})',
        gooey_options={
            'label': 'CPU Cores',
            'min': 1,
            'max': os.cpu_count()
        }
    )

    advanced.add_argument(
        '--max-memory-percent',
        metavar='Max Memory %',
        type=int,
        default=75,
        help='Maximum percentage of system memory to use',
        gooey_options={
            'min': 25,
            'max': 90
        }
    )

    advanced.add_argument(
        '--batch-size',
        metavar='Batch Size',
        type=int,
        default=2500,
        help='Number of photos to process in each batch (lower = less memory use but slower). Min 100, Max 5000',
        gooey_options={
            'min': 100,
            'max': 5000
        }
    )

    # Parse arguments
    args = parser.parse_args()

    # Then validate directories
    required_dirs = {
        'Metadata Directory': args.metadata_dir,
        'Photos Directory': args.photos_dir,
        'Output Directory': args.output_dir
    }

    missing_dirs = [name for name, path in required_dirs.items()
                    if path and not os.path.exists(os.path.dirname(path))]

    if missing_dirs:
        error_msg = f"Please set the following directories: {', '.join(missing_dirs)}"
        logging.error(error_msg)
        return args

    # Get the script's directory
    script_dir = Path(__file__).parent
    log_file = script_dir / 'flickr_to_immich.log'

    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create handlers
    file_handler = logging.FileHandler(log_file)  # Changed from args.log_file to log_file
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Set log level
    logging.root.setLevel(logging.INFO)

    logging.info("Logging initialized")

    try:
        # Handle preprocessing first
        if args.zip_preprocessing and args.source_dir:
            preprocessor = FlickrPreprocessor(
                source_dir=args.source_dir,
                metadata_dir=args.metadata_dir,
                photos_dir=args.photos_dir,
                quiet=args.quiet
            )
            preprocessor.process_exports()

        # Handle API key
        if args.use_api:
            if args.api_key:
                api_key = args.api_key
                os.environ['FLICKR_API_KEY'] = args.api_key
            else:
                api_key = os.environ.get('FLICKR_API_KEY')
            if not api_key:
                logging.warning("Flickr API enabled but no API key provided in GUI or environment")
        else:
            api_key = None

        # Create converter instance
        converter = FlickrToImmich(
            metadata_dir=args.metadata_dir,
            photos_dir=args.photos_dir,
            output_dir=args.output_dir,
            date_format=args.date_format,
            api_key=api_key,
            log_file=str(log_file),
            results_dir=args.results_dir,
            include_extended_description=not args.no_extended_description,
            write_xmp_sidecars=not args.no_xmp_sidecars,
            block_if_failure=args.export_block_if_failure,
            resume=args.resume,
            use_api=args.use_api,
            quiet=args.quiet,
            fave_weight=args.fave_weight,
            comment_weight=args.comment_weight,
            view_weight=args.view_weight,
            min_views=args.min_views,
            min_faves=args.min_faves,
            min_comments=args.min_comments,
            max_memory_percent=args.max_memory_percent,
            batch_size=args.batch_size,
            cpu_cores=args.cpu_cores
        )

        # Process based on export mode
        if args.export_mode == 'Full library and Highlights':
            logging.info("Processing both full library and highlights...")

            # Process library
            logging.info("Step 2: Processing full library...")
            print("\nProcessing full library...")
            converter.process_photos(args.organization, args.date_format)

            # Process highlights
            logging.info("Step 1: Creating interesting albums...")
            converter.create_interesting_albums(
                args.interesting_period,
                args.interesting_count
            )

            # Print statistics only once at the end
            converter.print_statistics()

        elif args.export_mode == 'Full library only':
            logging.info("Processing full library only...")
            print("\nProcessing full library...")
            converter.process_photos(args.organization, args.date_format)
            converter.print_statistics()

        elif args.export_mode == 'Highlights only':
            logging.info("Processing highlights only...")
            converter.create_interesting_albums(
                args.interesting_period,
                args.interesting_count)
            converter.print_statistics()

        # Print final statistics
        converter.write_results_log()

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.stdout.flush()
        raise

    return args

if __name__ == '__main__':
    print("Starting program...")
    print(f"CLI mode: {is_cli_mode()}")
    print(f"Arguments: {sys.argv}")

    if is_cli_mode():
        print("Entering CLI mode")
        sys.argv.remove('--cli')  # Remove CLI flag
        main()  # Run without Gooey
    else:
        print("Entering GUI mode")
        gui_main()  # Use the new gui_main function
