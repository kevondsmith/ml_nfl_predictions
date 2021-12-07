# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.0] - 2021-08-10

### Changed
- Removed local build files
- Removed soon to be deprecated setup.py `test` feature
- Synced tox and setup.py python versions
- Added new functions for pulling PFR, injury, snaps, NGS, and depth chart data
- Updated python requirements to >=py3.5
- Fixed bug in clean_nfl_data()
- Updated tests with new functions
- Update README to align with function updates

## [0.2.4] - 2021-08-28

### Changed
- Added feature  to cache play-by-play files locally
- Updated load_pbp_data() to work with locally cached files
- Updates README
