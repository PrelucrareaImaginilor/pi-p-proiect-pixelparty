# **WiDS Datathon++ 2025 University Challenge**

### Analiza literaturii de specialitate
| Nr. | Autori/An | Titlul articolului/proiectului | Aplicatie/Domeniu | Tehnologii utilizate | Metodologie/Abordare | Rezultate | Limitari | Comentarii suplimentare | 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. | L.K. Soumya Kumari, R. Sundarrajan / 2024 | A review on brain age prediction models | Brain age prediction, Deep learning | Mai multe tehnici de predictie utilizate in ultimii 11 ani pentru estimarea varstei creierului | Comparatie intre diverse tehnici si modele (SVR, CNNs, RNNs, combinatii de modele) | Cel mai bun rezultat obtinut pe fMRI-uri: SVR (Support Vector Regression)  MAE = 0.753 years |   | Un model de regresie eficient pentru datele proiectului nostru este SVR |
| 2. | Hongfang Han, Sheng Ge, Haixian Wang / 2023 | Prediction of brain age based on the community structure of functional networks | Brain age prediction | MATLAB | S-au testat 6 modele de machine learning diferite (SVR, RVR, LASSO, EN, RR, XGBoost) pentru predictia varstei | SVR MAE = 0.753 years | S-a folosit un numar mic de date, dintre care una singura pentru verificarea acuratetei finale |   |
| 3. | Siamak K. Sorooshyari / 2024 | Beyond network connectivity: A classification approach to brain age prediction with resting-state fMRI | Brain age/sex prediction | MATLAB, FMRIB Software Library | S-au utilizat fMRI-uri de la 887 de indivizi cu varste intre 21-85 ani, esantionate cu ajutorul metodei Monte-Carlo, pentru antrenarea unui SVM. Scopul a fost observarea acuratetei pentru categorii de varsta si diferentierea intre sexe. | S-a observat ca diferentele intre sexe se diminueaza odata cu cresterea in varsta devenind minime intre 61 si 70 de ani. S-a constatat ca nu se poate realiza o predictie a sexului individului. | Nu s-au realizat optimizari asupra spatiului de date | Avand in vedere ca datele primite pentru proiectul nostru consista in indivizi cu varste intre 5-22 ani, vom lua in considerare o esantionare pe sexe pentru o acuratete sporita |
| 4. | A | A | A | A | A | A | A | A |
| 5. | A | A | A | A | A | A | A | A |

### Arhitectura propusa
