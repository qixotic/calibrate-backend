# Calibrate Backend

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Backend for [Calibrate](https://calibrate.artpark.ai): Evaluation platform for **AI agents** (both text and voice agents).

## Installation

Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

## Running Locally

Start the development server:

```bash
cd src
uv run uvicorn main:app --reload
```

The app will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
