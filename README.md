# 5D/6D Entropická gravitace s PID regulací

## Systémový diagram
![PID Regulace](visuals/cosmic_pid_system.png)

### Popis
- **Modrý kruh**: 5D prostor (bulk)
- **Červená šipka**: Vstup entropie
- **Zelená šipka**: PID korekce

[Zobrazit plné rozlišení](visuals/cosmic_pid_system.png)



## Official Publication  
The full theory is published on Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15085762.svg)](https://doi.org/10.5281/zenodo.15085762)  

## How to Replicate Results  
- Install dependencies: `pip install -r requirements.txt`  
- Run `entropic_gravity_5D.ipynb` to reproduce Figure 2 from the paper.  


## 🚀 Quick Start
```bash
git clone https://github.com/marekzajda/5D_6D-theory-of-entropic-gravity.git
cd 5D_6D-theory-of-entropic-gravity
pip install -r requirements.txt
python holographic_bound.py
