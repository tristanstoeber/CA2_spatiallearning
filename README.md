# Silencing hippocampal CA2 reduces behavioral flexibility in spatial learning

## Quick-start
To reproduce the analysis clone the repo and run 
```
docker run -it -d -p 8888:8888 -e NB_USER=jovyan -e CHOWN_HOME=yes -e CHOWN_EXTRA_OPTS='-R' --user root -v /path/to/repo/:/home/jovyan/work/ tristanstoeber/ca2_spatiallearning jupyter-lab --ip 0.0.0.0 --allow-root --NotebookApp.token='hitti'
```
with `/path/to/repo/` adjusted to respective path.

## Data
Behavioral data are stored in `/data/expipe/` as an expipe project.
Data have been received as ANY-maze szd file.
After using the ANY-maze software to export to xml files, we used expipe to store experimental data together with metadata [[1]](#1).
If you use the data for further analysis, please also cite their original publication [[2]](#2).

## License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## References
<a id="1">[1]</a> 
https://expipe.readthedocs.io/en/latest/

<a id="2">[2]</a> 
Hitti & Siegelbaum, 2014, Nature
