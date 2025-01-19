**FLICKR EMMIGRATION TOOL**

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
**INTRODUCTION**

**What it does**

Takes a Flickr library and reformats it to various options to allow importing to other tools like Immich, PhotoPrism, Pixelfed, your own computer etc etc
It was written primarly to export to a folder format suitable for Immich and PixelFed
... but it gives generic outputs and lots of options and therefore is quite agnostic to the tool that is being used next. Yay for standards!

**Why did I build this?**

I am a 20 year user of Flickr. Flickr is a great tool for many reasons and serves a few core purposes:
1. It was orignally conceived for users to post and share their photos online with each other. It still does this fairly well (your mileage may vary here)
2. It allows for sharing photos with family and friends only (SO much better and more cleanly than Facebook or Instragram ever did)
3. It has become (for many) a place to just backup and not share photos.

Over time I have become disenchanted with Flickr
- The recent prevelance of AI generated content (ie not photos by photographers) is starting to take over the central feed
- It costs money to be part of the service. I always thought Flickr could be a forever home for my photos, but if you add this up over time it can get super expensive.
- I also dabbled with Instagram and had my account killed off with zero notice. I dislike the inability to control my own destiny, and therefore want more control.

I recently found the opensource solutions of Immich, PhotoPrism and Pixelfed, and I wanted a tool that would allow me to export my flickr data, then parse it to extract and maintain the metadata including albums, tags, favorites, comments etc.

So I wrote this (with some motivation from the Immich community)

**How it works**

At a high level the user flow is this:
1. Request export of flickr account
2. Unzip the files to folders (separate tool forthcoming to allow this to be made more autonomous)
3. Run the python script
4. The tool exports photos and videos to a new folder maintaining the flickr metadata in a format that is meaningful to you.

**Options of note**:

- Export by date, or by album name
- Highlights... create your own interestingness algorithm to allow you to export your top ranked images based on what you think is important.
- Embedding metadata in the photos EXIF description (e.g. flickr faves, comments etc)
- XMP sidecar files that contain the additional metadata that cannot be stored in a photos EXIF metadata
- Resume function

----------------------------
**USAGE**

(First Download the files. Go to Green <> Code button and download the zip file to your computer)

1. **Request an export of your Flickr data**

  - go to https://www.flickr.com/account then under "Your Flickr Data" request it.
  - After a week or so flickr will email you with a link to the data.
  - NOTE - BEFORE you run the script on your massive library you may want to create a test flickr account, add a few images to that, favorite them, add to albums, comment etc. Then download that. With a small account it can take just a few hours to request your flickr archive. With a large account it can take over a week!

2. **Download ALL of the zip files**.

Depending on how large your account is you may have hundreds of zip files! Note there is one file for account data, and one for photos and videos. Download them all!

3. On your computer where you will be running the script, **create the following folders**

   metadata, photos, output, results

   File and Folder Structure Setup:

    Example folder and script structure is as follows. Note these locations can be anywhere for each folder. You will pass the locations into the script when you run it.
    So for example you can have metadata, photos and output folders in different locations (e.g. on your NAS), and the script folder on your desktop computer. Your call.

    - root folder (anywhere on your system e.g. in home folder):
        - folder: metadata
        - folder: photos
        - folder: results
        - folder (will be created automatically by this script if not already existing): output
        - files: script files including:
                - flickr-to-immich.py
                - flickr-to-folders.py
                - package_dependencies.txt

    Extract the zip files accordingly:
    - for the account data zip extract the contents into 'metadata'
    - for the photo AND video files extract the contents into 'photos'

4. **Set up your environment - dependencies**:

    - System dependencies: you must have:
       Python >= 3.6 (for f-strings, Path objects)

    - a) Initialize Flickr API
        i) Get your API Key:
                - Go to https://www.flickr.com/services/api/, then select 'API Keys'.
                - Apply for a non commercial key. Fill out the form and just state that e.g.
                    " I am applying for a key to use an opensource script that access additional metadata missing from downloaded flickr archives.
                    This will allow me to keep an offline copy of my images in my own backup tool with more complete metadata'
        ii) Run the specific command for your OS, and pass in your API Key
            # Linux/Mac
            export FLICKR_API_KEY='your_api_key_here'

            # Windows CMD
            set FLICKR_API_KEY=your_api_key_here

            # Windows PowerShell
            $env:FLICKR_API_KEY = 'your_api_key_here'

    - b) Ensure you have dependencies installed. See dependencies for more detail.
            Tools
                - exiftool (Must be installed and available in system PATH)
                    - On Ubuntu/Debian: sudo apt-get install libimage-exiftool-perl
                    - On MacOS: brew install exiftool
                    - On Windows: Download from https://exiftool.org and add to PATH-
                - Packages
                    - Either:
                        pip install -r package_dependencies.txt
                    - Or if on Arch based linux distro, or you want to keep things clean you may want to consider setting up virtual environment:
                        # Create a virtual environment in your project directory
                        python -m venv .venv

                        # Activate the virtual environment
                        source .venv/bin/activate

                        # Now install the dependencies in the virtual environment
                        pip install -r package_dependencies.txt

5. **Set up envirnoment and run the script**:
    - First open your terminal and navigate to the folder where your script is stored
    - if you are using a virtual environment (see Flickr API section above), you must first activate the virtual env before running the above line:
        i) Run this - a one time setup for the directory you are running the script in:
          python -m venv .venv
        ii) then you'll need to do this every time you open a new terminal:
          source .venv/bin/activate

    - Install dependencies:
        pip install -r package_dependencies.txt

    - Initialize flickr API
        - See above for details. e.g. on Linux:
        export FLICKR_API_KEY='your_api_key_here'

    - Run it:
        - NOTE: change the path in front of metadata, photos and output to correspond to where you want these folders to be.
            e.g. instead of ./ (which would be in the same directory as the script, remove the . and change it to something like '/folder_location_on_my_nas/metadata'

        (run this command to see all options. These are also described below in OPTIONAL FLAG USAGE:
            python flickr-to-immich.py --help
        )

        example:
        python flickr-to-immich.py --metadata-dir "./metadata" --photos-dir "./photos" --output-dir "./output" --organization by_album --interesting-period all-time --interesting-count 100

        **OPTIONAL FLAG USAGE**:

        - Two main options: Export full library or just highlights ()
           (If no flag set, then export both highlights and full library)
           --export-interesting-only (only export the highlights)
           --export-standard-only (exports the full library)

          - Full library: folder output format. You have two options:

              i) output flickr ALBUMS as folders. Do this if you want to use e.g. the Immich CLI import tool to load as albums.
                  WARNING: as Flickr allows you assign 1 image to many albums you may end up with duplicates in your library!
                  If like me and you tend to only assign 1 image to 1 album this will be fine and very much simplify library import to the tool of your choice

                  by album, set flag to  --organization by_album
                      example:
                      python flickr-to-immich.py --metadata-dir "./metadata" --photos-dir "./photos" --output-dir "./output" --organization by_album

              ii) output into folders by DATE CREATED. Do this if you don't want any duplicate images created.
                  WARNING: Your albums will only live as attributes in the XMP metadata and image EXIF in the description (should you choose to show the additional info there).
                  So if you are really worried about duplicates just use this option and then search the image description or metadata in the sidecar file to re-create your albums.
                  ... but that will be rather labor intensive!
                  (Note if you use Immich it doesn't support XMP metadata fully yet.)

                  by date, set flag to --date-format <date format>
                      e.g.
                          --organization by_date --date-format yyyy/yyyy-mm-dd
                          --organization by_date --date-format yyyy/yyyy-mm
                          --organization by_date --date-format yyyy-mm

          - For creating 'My Highlights' folders automatically. This is sort of like flickr interestingness, but uses your own algorithm because Flickr's interestingness score is a black box

              This is useful if you want to show the most interesting photos from your library. For example maybe you only want to export those? Maybe you want to just load these to a sharing tool like PixelFed and load all the images to Immich? Your call!

              To change the scoring, scroll down to interestingness_score and make your algorithim reflect what you think is important!
              (Note I called it interestingness_score becaused I originally set out to use flickr's interestigness method. I ran into issues, so instead made this approach
              If you want me to implement the actual interestigness score from flickr, please add a feature request.)

              default: interestingness_score = views + (faves * 10) + (comments * 5)
              also not the cut-offs for what is interesting are as follows. To adjust them search for that in the code below

              search for this and modify!
              if faves > 0 or comments > 0 or views >= 3:

              an integer number between 1 and the max number of images that will be exported as interesting.

              Highest engagement gets rank 1
              Lowest engagement gets rank N

              NOTE: You can opt to upload the highlights also to your tookl of choice. However note that it isn't the full library.
              So why did I write this? So that I can create a folder of my best images, and upload them to a specific folder in Immich, or another more community focussed sharing tool like Pixelfed.
              ... after all Immich (or PhotoPrism) are really tools for backing up and accessing your photos. They aren't really a photo platform like flickr is.
              ... Flickr serves two purposes: 1) originally intended for community sharing; 2) backing up photos also

              usage guide:
              For all-time interesting photos set the --interesting-period flag to 'all-time'
              For a interesting photo folders for images captured by year set the --interesting-period flag to 'byyear'
              To set the number of images in the interestingness albums, set the --interesting-count flag e.g. 100

        - For exporting XMP Sidecars and embedding flickr metadata into the photo's EXIF description field:
            - Use defaults (both enabled)- no flags to add
            - Disable extended description add --no-extended-description
            - Disable XMP sidecars add --no-xmp-sidecars

        - Resume: If you had to stop processing, you can resume the script to process files not already processed:
            --resume

        - Block files from exporting:  Block file export if metadata or file processing fails
            Add this flag only if you want it block export if metadata or file processing fails. Leave it out and it will export the photo regardless of the failure.
            --export-block-if-failure

        - Logging verbosity:
            Add this flag to make the console output just show status bar while running. The log files will show the errors:
            --quiet


6. **Import to your tool of choice**

  e.g. for Immich use the CLI tool:
    - Must use Immich CLI in order to maintain the albums (as folders) and XMP sidecar files containing additional metadata not supported in the standard EXIF metadata that is supported in the image file itself.
    - Read these instructions here https://immich.app/docs/features/command-line-interface
