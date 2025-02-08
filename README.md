**FLICKR to ANY TOOL**

---------------------------
by Rob Brown

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

---------------------------
## Introduction

### What it does

Takes a Flickr library and reformats it to various options to allow importing to other tools like Immich, PhotoPrism, Pixelfed, your own computer, etc. It was written primarily to export to a folder format suitable for Immich and PixelFed, but it gives generic outputs and lots of options and therefore is quite agnostic to the tool that is being used next.

### Key Features

- User-friendly GUI interface for easy operation
- Command-line interface for automation and scripting
- Export by date or album organization
- Highlights feature to identify your most engaging photos
- Embedded EXIF metadata preservation
- XMP sidecar files with extended metadata
- Resume functionality for interrupted exports

## Installation

### Prerequisites

- Python 3.6 or higher
- System Dependencies:
  - ExifTool (must be in system PATH)
    - Ubuntu/Debian: `sudo apt-get install libimage-exiftool-perl`
    - MacOS: `brew install exiftool`
    - Windows: Download from [ExifTool website](https://exiftool.org) and add to PATH

### Installation Methods

#### Option 1: Using pip (Recommended)

Navigate to the project directory where you want to store the script, then open a terminal window and type:

```bash
pip install flickr-to-anytool
```

#### Option 2: Installing from Source

1. Clone or download the repository
2. Navigate to the project directory
3. Create virtual environment in your project directory (recommended)
```bash
python -m venv .venv
```
Then you must activate the virtual environment:
```bash
source .venv/bin/activate
```
4. Install using pip:
```bash
pip install .
```

For development installation (editable mode):
```bash
pip install -e .
```

This will automatically install all required dependencies defined in setup.py.

## Usage

### Step 1: Request Your Flickr Data

1. Go to [Flickr Account Settings](https://www.flickr.com/account)
2. Under "Your Flickr Data", request your data export
3. Wait for email notification (can take several days for large accounts)
4. Download all provided zip files

### Step 2: Directory Setup

Create these directories (can be anywhere on your system):
- metadata
- photos
- output
- results

### Step 3: Flickr API Setup (Optional but Recommended)

1. Get your API Key from [Flickr API](https://www.flickr.com/services/api/)
2. Set the environment variable:
```bash
# Linux/Mac
export FLICKR_API_KEY='your_api_key_here'

# Windows CMD
set FLICKR_API_KEY=your_api_key_here

# Windows PowerShell
$env:FLICKR_API_KEY = 'your_api_key_here'
```

### Step 4: Run the Tool

First navigate to the the folder where you installed the script. Open a terminal for that folder. Or if on windows / mac, open terminal / prompt and then navigate to that folder.

#### !! If you have logged out recently, do these steps first:

navigate to your script directory in terminal.

Then:
```bash
source .venv/bin/activate
```

Then activate the flickr API
```bash
export FLICKR_API_KEY='your_api_key_here'
```

#### Option 1: GUI Mode (Recommended for most users)

Run:
```bash
flickr-to-any
```

The GUI provides an intuitive interface with:
- Step-by-step workflow
- Directory selection dialogs
- Export type options
- Advanced settings configuration
- Progress visualization

#### Option 2: CLI Mode (For automation/scripting)

(optional) If you made any changes to the script:
```bash
pip install -e .
```

Then run the script:

Basic usage:
```bash
flickr-to-any-cli --metadata-dir "./metadata" --photos-dir "./photos" --output-dir "./output"
```

Common options:
```bash
# Export with zip preprocessing
flickr-to-any-cli --zip-preprocessing --source-dir "./source" --metadata-dir "./metadata" --photos-dir "./photos" --output-dir "./output"

# Export by album organization
flickr-to-any-cli --organization by_album --date-format "yyyy/yyyy-mm-dd"

# Export highlights only
flickr-to-any-cli --export-interesting-only --interesting-period all-time --interesting-count 100
```

## Organization Options

### 1. By Album (`--organization by_album`)
- Creates folders matching your Flickr albums
- Note: May create duplicates if photos are in multiple albums
- Unorganized photos go to "00_NoAlbum" with date subfolders

### 2. By Date (`--organization by_date`)
- Organizes photos in date-based folders
- No duplicates
- Album information preserved in metadata

### 3. Hybrid (`--organization hybrid`)
- Date-based folders with album subfolders
- Best of both worlds but may create duplicates

## Highlights Feature

Creates curated collections of your most engaging photos:
- Organized by privacy level (Public, Private, Friends, Family)
- Customizable engagement scoring
- Useful for creating showcase collections

### Customizing the Engagement Score

You can fine-tune how "interesting" photos are selected using these parameters:

#### Weight Multipliers
- `--fave-weight` (default: 10.0): How much to multiply favorites by
- `--comment-weight` (default: 5.0): How much to multiply comments by
- `--view-weight` (default: 0.1): How much to multiply views by

#### Minimum Thresholds
- `--min-views` (default: 20): Minimum views required
- `--min-faves` (default: 0): Minimum favorites required
- `--min-comments` (default: 0): Minimum comments required

Example CLI usage:
```bash
flickr-to-any-cli --fave-weight 15.0 --comment-weight 7.5 --view-weight 0.2 --min-views 30 --min-faves 1
```

In GUI mode, these parameters can be adjusted in the Export Type section. The final interestingness score for each photo is calculated as:

```
score = (favorites × fave_weight) + (comments × comment_weight) + (views × view_weight)
```

Photos must meet at least one of the minimum thresholds (views, favorites, or comments) to be considered for highlights.

## Advanced Features

- **XMP Sidecars**: Comprehensive metadata in separate files
- **Extended EXIF**: Embedded Flickr metadata in photos
- **Resume Support**: Continue interrupted exports
- **API Integration**: Enhanced metadata retrieval
- **Custom Processing**: Flexible organization options

## Importing to Other Tools

### Immich
Use the [Immich CLI](https://immich.app/docs/features/command-line-interface) for best results:
- Preserves album structure
- Maintains XMP sidecars
- Retains extended metadata

### Other Tools
The organized output is compatible with:
- PhotoPrism
- Pixelfed
- Standard file systems
- Most photo management tools

## Troubleshooting

Common issues and solutions:
1. **Missing Dependencies**: Run `python setup.py install` to properly install all dependencies
2. **ExifTool Errors**: Ensure ExifTool is properly installed and in PATH
3. **API Key Issues**: Check environment variable setup

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

GNU General Public License v3.0 - see LICENSE file for details

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/brownphotographic/Flickr2Any-Tool/issues)
