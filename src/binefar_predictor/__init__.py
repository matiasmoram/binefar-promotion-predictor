"""binefar_predictor — will CD Binéfar be promoted next season?

An end-to-end statistical pipeline:

    scrape (Sofascore) -> assemble matches -> fit Dixon-Coles / Elo ratings
      -> Monte-Carlo simulate the season -> promotion probability -> calibrate

See ``README.md`` for the methodology and data-source notes.
"""

__version__ = "1.0.0"
