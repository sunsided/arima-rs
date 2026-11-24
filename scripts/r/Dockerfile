# The R environment for `scripts/generate_r_reference.R`.
# rocker/r-ver pins R and installs CRAN packages from a dated Posit snapshot.
FROM rocker/r-ver:4.4.2
RUN install2.r --error --skipinstalled forecast
WORKDIR /work
