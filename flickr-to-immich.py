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
import flickrapi
import xml.etree.ElementTree as ET
from importlib import metadata
from tomlkit.api import key
from PIL import Image

class MediaType(Enum):
    """Supported media types"""
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"

# Configuration flags
INCLUDE_EXTENDED_DESCRIPTION = True  # Set to False to only include original description
WRITE_XMP_SIDECARS = True  # Set to False to skip writing XMP sidecar files

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

    def _build_photo_album_mapping(self):
            """Build mapping of photos to their albums"""
            try:
                self.photo_to_albums = {}  # Initialize the dictionary
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
                logging.error(f"Error building photo-album mapping: {str(e)}")
                self.photo_to_albums = {}  # Initialize empty if there's an error
                raise

    def __init__(self,
                metadata_dir: str,  # Directory containing JSON files
                photos_dir: str,    # Directory containing media files
                output_dir: str,    # Directory for album structure output
                api_key: Optional[str] = None,
                log_file: Optional[str] = None,
                results_dir: Optional[str] = None,  # Add this new parameter
                include_extended_description: bool = INCLUDE_EXTENDED_DESCRIPTION,
                write_xmp_sidecars: bool = WRITE_XMP_SIDECARS,
                resume: bool = False):

        self.resume = resume

        # Track processing statistics
        self.stats = {
            'total_files': 0,
            'successful': {
                'count': 0
            },
            'failed': {
                'count': 0,
                'details': []  # Will store tuples of (image_file, json_file, error_msg)
            },
            'skipped': {
                'count': 0,
                'details': []  # Will store tuples of (image_file, json_file, skip_reason)
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
        """
        self.include_extended_description = include_extended_description
        self.write_xmp_sidecars = write_xmp_sidecars

        # Setup logging
        self._setup_logging(log_file)

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

    def _setup_logging(self, log_file: Optional[str]):
        """Configure logging with both file and console output"""
        # Delete existing log file if it exists
        if log_file and Path(log_file).exists():
            try:
                Path(log_file).unlink()
            except Exception as e:
                print(f"Warning: Could not delete existing log file: {e}")

        # Set up format for logging
        log_format = '%(levelname)s - %(message)s'

        # Set up console handler with minimal output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)  # Only show warnings and errors in console
        console_handler.setFormatter(logging.Formatter(log_format))

        # Configure root logger
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)

        # Set up file handler if log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # More detailed logging in file
            file_handler.setFormatter(logging.Formatter('%(asctime)s - ' + log_format))
            logging.getLogger().addHandler(file_handler)

        # Reduce logging from external libraries
        logging.getLogger('flickrapi').setLevel(logging.ERROR)
        logging.getLogger('requests').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)

    def write_results_log(self):
        """Write a detailed results log file"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.results_dir / 'processing_results.txt'

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("FLICKR TO IMMICH PROCESSING RESULTS\n")
                f.write("=================================\n\n")

                # Write timestamp
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Write summary counts
                f.write("SUMMARY\n-------\n")
                f.write(f"Total files processed: {self.stats['total_files']}\n")
                f.write(f"Successfully processed: {self.stats['successful']['count']}\n")
                f.write(f"Failed to process: {self.stats['failed']['count']}\n")
                f.write(f"Skipped due to resume function: {self.stats['skipped']['count']}\n\n")

                # Write failed files section
                f.write("FAILED FILES\n------------\n")
                if self.stats['failed']['count'] == 0:
                    f.write("No failed files\n")
                else:
                    for img_file, json_file, error in self.stats['failed']['details']:
                        f.write(f"\nImage File: {img_file}\n")
                        f.write(f"JSON File: {json_file}\n")
                        f.write(f"Error: {error}\n")
                        f.write("-" * 50 + "\n")

                # Write skipped/not processed files section
                f.write("\nSKIPPED due to resume\n-------------------------\n")
                if self.stats['skipped']['count'] == 0:
                    f.write("No skipped files\n")
                else:
                    for img_file, json_file, reason in self.stats['skipped']['details']:
                        f.write(f"\nImage File: {img_file}\n")
                        f.write(f"JSON File: {json_file}\n")
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

            # Create parent metadata
            parent_metadata = {
                'title': "My Flickr Highlights",
                'description': "Collection of your most engaging Flickr photos",
                'created': datetime.now().isoformat(),
                'flickr_source': 'user_engaging'
            }
            self._save_album_metadata(highlights_dir, parent_metadata)

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

            album_metadata = {
                'title': album_name,
                'description': description,
                'created': datetime.now().isoformat(),
                'flickr_source': 'user_engaging',
                'photo_count': len(photos)
            }
            self._save_album_metadata(album_dir, album_metadata)

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

            # Create album directories under by_album
            for album in self.albums:
                album_dir = full_export_dir / self._sanitize_folder_name(album['title'])
                album_dir.mkdir(parents=True, exist_ok=True)

                # Save album metadata
                self._save_album_metadata(album_dir, album)

            logging.info(f"Created {len(self.albums)} album directories under by_album")
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

    def _save_album_metadata(self, album_dir: Path, album: Dict):
        """Save album metadata to a hidden file"""
        try:
            metadata = {
                'title': album.get('title', ''),
                'description': album.get('description', ''),
                'created': album.get('created', ''),
                'last_updated': album.get('last_updated', ''),
                'flickr_id': album.get('id', ''),
                'flickr_url': album.get('url', '')
            }

            meta_file = album_dir / '.album_metadata'
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save metadata for album {album.get('title', 'Unknown')}: {str(e)}")

    def _sanitize_folder_name(self, name: str) -> str:
        """Convert album name to safe folder name"""
        # Replace spaces with underscores and remove special characters
        sanitized = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                         for c in name.replace(' ', '_'))
        return sanitized.strip('_')

    def process_photos(self, organization: str, date_format: str = None):
        """Process all photos: copy to albums and embed metadata"""
        self.stats['total_files'] = len(self.photo_to_albums)

        # Create progress bar
        with tqdm(total=self.stats['total_files'], desc="Processing photos", leave=True) as pbar:
            for photo_id, albums in self.photo_to_albums.items():
                try:
                    if organization == 'by_album':
                        success = self._process_single_photo_by_album(photo_id, albums)
                    else:  # by_date
                        success = self._process_single_photo_by_date(photo_id, date_format)

                    if success:
                        self.stats['successful']['count'] += 1
                    pbar.update(1)
                except Exception as e:
                    self.stats['failed']['count'] += 1
                    self.stats['failed']['details'].append(
                        (f"unknown_{photo_id}", f"photo_{photo_id}.json", str(e))
                    )
                    logging.error(f"Error processing {photo_id}: {str(e)}")


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
        """Process a single photo for album-based organization"""
        try:
            # Load photo metadata
            photo_json = self._load_photo_metadata(photo_id)
            if not photo_json:
                self.stats['skipped']['count'] += 1
                self.stats['skipped']['details'].append(
                    (f"unknown_{photo_id}", f"photo_{photo_id}.json", "Metadata file not found")
                )
                return False

            # Find the source file
            source_file = self._find_photo_file(photo_id, photo_json['name'])
            if not source_file:
                self.stats['skipped']['count'] += 1
                self.stats['skipped']['details'].append(
                    (photo_json['name'], f"photo_{photo_id}.json", "Source file not found")
                )
                return False

            # Get clean original filename from metadata
            original_name = photo_json['name']
            source_extension = source_file.suffix
            if not original_name.lower().endswith(source_extension.lower()):
                original_name = f"{original_name}{source_extension}"

            media_type = self.get_media_type(source_file)
            if media_type == MediaType.UNKNOWN:
                logging.warning(f"Unsupported media type for {source_file}")
                self.stats['skipped']['count'] += 1
                return False

            # Track if we need to process any albums
            needs_processing = False

            # Check each album
            for album_name in album_names:
                album_dir = self.output_dir / "full_library_export" / "by_album" / self._sanitize_folder_name(album_name)
                dest_file = album_dir / original_name

                # Skip if file exists and we're resuming
                if self.resume and dest_file.exists():
                    logging.debug(f"Skipping existing file: {dest_file}")
                    continue

                needs_processing = True
                break

            # If resuming and file exists in all albums, skip processing
            if self.resume and not needs_processing:
                logging.debug(f"Skipping {photo_id} - exists in all albums")
                self.stats['skipped']['count'] += 1
                return True

            # Process the photo for each album
            for album_name in album_names:
                album_dir = self.output_dir / "full_library_export" / "by_album" / self._sanitize_folder_name(album_name)
                album_dir.mkdir(parents=True, exist_ok=True)
                dest_file = album_dir / original_name

                # Skip if file exists and we're resuming
                if self.resume and dest_file.exists():
                    continue

                logging.debug(f"Copying {source_file} to {dest_file}")
                shutil.copy2(source_file, dest_file)

                # Embed metadata based on media type
                if media_type == MediaType.IMAGE:
                    self._embed_image_metadata(dest_file, photo_json)
                    if self.write_xmp_sidecars:
                        self._write_xmp_sidecar(dest_file, photo_json)
                elif media_type == MediaType.VIDEO:
                    self._embed_video_metadata(dest_file, photo_json)
                    if self.write_xmp_sidecars:
                        self._write_xmp_sidecar(dest_file, photo_json)

            # Update success counter
            self.stats['successful']['count'] += 1
            return True

        except Exception as e:
            error_msg = f"Error processing {photo_id}: {str(e)}"
            self.stats['failed']['count'] += 1
            self.stats['failed']['details'].append(
                (str(source_file) if 'source_file' in locals() else f"unknown_{photo_id}",
                f"photo_{photo_id}.json",
                error_msg)
            )
            logging.error(error_msg)
            return False

    def _process_single_photo_by_date(self, photo_id: str, date_format: str) -> bool:
        """Process a single photo for date-based organization"""
        try:
            # Load photo metadata
            photo_json = self._load_photo_metadata(photo_id)
            if not photo_json:
                self.stats['skipped']['count'] += 1
                self.stats['skipped']['details'].append(
                    (f"unknown_{photo_id}", f"photo_{photo_id}.json", "Metadata file not found")
                )
                return False

            # Find the source file
            source_file = self._find_photo_file(photo_id, photo_json['name'])
            if not source_file:
                self.stats['skipped']['count'] += 1
                self.stats['skipped']['details'].append(
                    (photo_json['name'], f"photo_{photo_id}.json", "Source file not found")
                )
                return False

            # Get date taken from metadata
            date_taken = photo_json.get('date_taken')
            if not date_taken:
                self.stats['skipped']['count'] += 1
                self.stats['skipped']['details'].append(
                    (str(source_file), f"photo_{photo_id}.json", "No date information available")
                )
                return False

            # Get clean original filename from metadata
            original_name = photo_json['name']
            source_extension = source_file.suffix
            if not original_name.lower().endswith(source_extension.lower()):
                original_name = f"{original_name}{source_extension}"

            # Create date-based directory path
            date_path = self._get_date_path(date_taken, date_format)
            date_dir = self.output_dir / "full_library_export" / "by_date" / date_path
            dest_file = date_dir / original_name

            # Skip if file exists and we're resuming
            if self.resume and dest_file.exists():
                logging.debug(f"Skipping existing file: {dest_file}")
                self.stats['skipped']['count'] += 1
                return True

            media_type = self.get_media_type(source_file)
            if media_type == MediaType.UNKNOWN:
                logging.warning(f"Unsupported media type for {source_file}")
                self.stats['skipped']['count'] += 1
                return False

            # Create directory and copy file
            date_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Copying {source_file} to {dest_file}")
            shutil.copy2(source_file, dest_file)

            # Embed metadata based on media type
            if media_type == MediaType.IMAGE:
                self._embed_image_metadata(dest_file, photo_json)
                if self.write_xmp_sidecars:
                    self._write_xmp_sidecar(dest_file, photo_json)
            elif media_type == MediaType.VIDEO:
                self._embed_video_metadata(dest_file, photo_json)
                if self.write_xmp_sidecars:
                    self._write_xmp_sidecar(dest_file, photo_json)

            # Update success counter
            self.stats['successful']['count'] += 1
            return True

        except Exception as e:
            error_msg = f"Error processing {photo_id}: {str(e)}"
            self.stats['failed']['count'] += 1
            self.stats['failed']['details'].append(
                (str(source_file) if 'source_file' in locals() else f"unknown_{photo_id}",
                f"photo_{photo_id}.json",
                error_msg)
            )
            logging.error(error_msg)
            return False

    def _embed_image_metadata(self, photo_file: Path, metadata: Dict):
        """Embed metadata into an image file using exiftool with enhanced error handling"""
        try:
            # Start with basic, safe metadata arguments
            args = [
                'exiftool',
                '-overwrite_original',
                '-ignoreMinorErrors',
                '-m',
                '-P',  # Preserve existing metadata
                '-E',  # Extract embedded metadata

                # Core metadata that's less likely to cause issues
                f'-Description={metadata.get("description", "")}',
                f'-Title={metadata.get("name", "")}',
                f'-Author={self.account_data.get("real_name", "")}',
                f'-Copyright={metadata.get("license", "All Rights Reserved")}',

                # Handle date separately to avoid format issues
                f'-DateTimeOriginal={metadata.get("date_taken", "")}',
            ]

            # Add keywords/tags safely
            for tag in metadata.get('tags', []):
                if isinstance(tag, dict) and 'tag' in tag:
                    args.append(f'-Keywords={tag["tag"]}')

            # Handle GPS data carefully
            if metadata.get('geo'):
                geo = metadata['geo']
                if all(key in geo and isinstance(geo[key], (int, float))
                    for key in ['latitude', 'longitude']):
                    args.extend([
                        f'-GPSLatitude={geo["latitude"]}',
                        f'-GPSLongitude={geo["longitude"]}',
                    ])

            # Add the enhanced description using a more reliable field
            enhanced_description = self._build_formatted_description(metadata)
            args.extend([
                f'-ImageDescription={enhanced_description}',
                f'-IPTC:Caption-Abstract={enhanced_description}'
            ])

            # Add the target file at the end
            args.append(str(photo_file))

            # First, try to read existing metadata
            check_args = ['exiftool', '-j', str(photo_file)]
            try:
                result = subprocess.run(check_args, capture_output=True, text=True)
                if result.returncode == 0:
                    logging.debug(f"Successfully read existing metadata from {photo_file}")
                else:
                    logging.warning(f"Warning reading metadata from {photo_file}: {result.stderr}")
            except Exception as e:
                logging.warning(f"Error checking existing metadata: {str(e)}")

            # Now try to write the new metadata
            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)

            if result.stderr:
                # Log warnings but don't fail unless it's a critical error
                if "Bad format" in result.stderr or "Suspicious IFD0" in result.stderr:
                    logging.warning(f"Non-critical EXIF warning for {photo_file}: {result.stderr}")
                else:
                    logging.debug(f"Exiftool message for {photo_file}: {result.stderr}")

        except subprocess.CalledProcessError as e:
            if "Bad format" in str(e.stderr) or "Suspicious IFD0" in str(e.stderr):
                # For known EXIF issues, try a more conservative approach
                try:
                    conservative_args = [
                        'exiftool',
                        '-overwrite_original',
                        '-ignoreMinorErrors',
                        '-m',
                        f'-ImageDescription={enhanced_description}',
                        f'-IPTC:Caption-Abstract={enhanced_description}',
                        str(photo_file)
                    ]
                    result = subprocess.run(conservative_args, capture_output=True, text=True)
                    if result.returncode == 0:
                        logging.info(f"Successfully embedded basic metadata in {photo_file} using conservative approach")
                        return
                except Exception as inner_e:
                    logging.error(f"Error in conservative metadata embedding for {photo_file}: {str(inner_e)}")
                    raise

            error_msg = f"Error embedding metadata in {photo_file}: {e.stderr}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error embedding metadata in {photo_file}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _load_photo_metadata(self, photo_id: str) -> Optional[Dict]:
        """Load metadata for a specific photo"""
        try:
            # Try different possible metadata file patterns
            possible_patterns = [
                f"photo_{photo_id}.json",
                f"photo_{int(photo_id):d}.json",  # Handle numerical IDs
                f"photo_{photo_id.lstrip('0')}.json"  # Handle IDs with leading zeros
            ]

            for pattern in possible_patterns:
                photo_file = self.metadata_dir / pattern
                if photo_file.exists():
                    with open(photo_file, 'r', encoding='utf-8') as f:
                        return json.load(f)

            logging.error(f"Metadata file not found for photo {photo_id}")
            return None

        except Exception as e:
            logging.error(f"Error loading metadata for photo {photo_id}: {str(e)}")
            return None

    def _find_photo_file(self, photo_id: str, filename: str) -> Optional[Path]:
        """Find the original photo file using the Flickr ID"""
        # Try exact match first
        for file in self.photos_dir.iterdir():
            if f"_{photo_id}_" in file.name:
                return file

        # If not found, try normalized versions (handle old-style IDs)
        normalized_id = photo_id.lstrip('0')  # Remove leading zeros
        for file in self.photos_dir.iterdir():
            file_parts = file.name.split('_')
            for part in file_parts:
                if part.lstrip('0') == normalized_id:
                    return file

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
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous run - skip existing files and preserve output directory')

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
            resume=args.resume
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
