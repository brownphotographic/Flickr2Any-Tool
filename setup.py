from setuptools import setup, find_packages

setup(
    name="flickr-to-anytool",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flickrapi',
        'Pillow',
        'tqdm',
        'wxPython',
        'Gooey',
        'tomlkit'
        'psutil',
    ],
    entry_points={
        'console_scripts': [  # Changed from gui_scripts to console_scripts
            'flickr-to-any=flickr_to_anytool.flickr_to_any:gui_main',
        ],
    },
)
