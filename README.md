## purpose
- fine‑tune an off‑the‑shelf en→fr mt model on medical‑domain data.
- goal: beat the base model on automatic metrics and produce cleaner, more domain‑appropriate phrasing.

## results (benchmarks)
- bleu: 33.0 → 49.7 (+16.7)
- meteor: 0.527 → 0.652 (+0.125)
- bertscore f1: 0.824 → 0.871 (+0.047)

what this means
- stronger lexical choices for medical/legal phrasing; fewer literal errors.
- base sometimes picks incorrect words (e.g., "dépassés"); sft corrects to domain terms ("apurés").
- remaining issues: occasional odd acronyms and rare truncations; mitigations include length‑penalty tuning, more in‑domain data, and longer training.

## examples (base vs sft vs reference)

example 0
```
english      : Figure 1: Measles Vaccination Coverage Rates in the EU in 2016 (Source: WHO/UNICEF JRF)
sft prediction: Graphique 1: Taux de couverture vaccinale de la rougeole dans l'UE en 2016 (Source: CJJ de l'OMS/UNICEF)
base prediction: Figure 1 : Taux de couverture vaccinale de la rougeole dans l'UE en 2016 (Source : OMS/UNICEF JRF)
reference    : Figure 1: Taux de couverture vaccinale contre la rougeole dans l'UE en 2016 (Source: Formulaire commun OMS/Unicef de notification sur la vaccination)
```

example 1
```
english      : ][8: Commission Directive 2006/17/EC of 8 February 2006 implementing Directive 2004/23/EC of the European Parliament and of the Council as regards certain technical requirements for the donation, procurement and testing of human tissues and cells (OJ L38, 9.2.2006, p. 40).
sft prediction: ][8: Directive 2006/17/CE de la Commission du 8 février 2006 portant application de la directive 2004/23/CE du Parlement européen et du Conseil en ce qui concerne certaines exigences techniques relatives au don, à l'obtention et au test de tissus et cellules d'origine humaine (JO L 38 du 9.2.2006, p. 40).
base prediction: [8: directive 2006/17/CE de la Commission du 8 février 2006 portant application de la directive 2004/23/CE du Parlement européen et du Conseil en ce qui concerne certaines exigences techniques pour le don, l'acquisition et l'essai de tissus et cellules humains (JO L 38 du 9.2.2006, p. 40).
reference    : ][8: Directive 2006/17/CE de la Commission du 8 février 2006 portant application de la directive 2004/23/CE du Parlement européen et du Conseil concernant certaines exigences techniques relatives au don, à l'obtention et au contrôle de tissus et de cellules d'origine humaine (JO L 38 du 9.2.2006, p. 40).
```

example 2
```
english      : The non-recovered amounts in 2020 will be cleared at programme closure.
sft prediction: Les montants non recouvrés en 2020 seront apurés lors de la clôture du programme.
base prediction: Les montants non recouvrés en 2020 seront dépassés au moment de la clôture du programme.
reference    : Les montants non récupérés en 2020 seront apurés à la clôture des programmes.
```

example 3
```
english      : We need to protect workers from unemployment and loss of income where possible, as they should not become victim of the outbreak.
sft prediction: Nous devons protéger les travailleurs contre le chômage et la perte de revenus lorsque cela est possible, car ils ne devraient pas être victimes de l'épidémie.
base prediction: Nous devons protéger les travailleurs contre le chômage et la perte de revenus dans la mesure du possible, car ils ne devraient pas être victimes de l'épidémie.
reference    : Nous devons protéger les travailleurs du chômage et de la perte de revenu lorsque cela est possible, pour ne pas en faire des victimes collatérales de l'épidémie.
```

example 4
```
english      : ...
sft prediction: La Commission organisera une conférence européenne sur la question de savoir comment.
base prediction: La Commission organisera une conférence européenne en vue de recueillir des informations.
reference    : La Commission organisera une conférence européenne avant la fin de 2016 afin de recueillir des informations en retour.
```

highlights from examples
- sft fixes key domain terms (ex2: "apurés" vs base "dépassés").
- sft leans toward legalese formulations (ex1: "obtention"), closer to the reference style.
- base can be more conservative with punctuation/formatting; sft occasionally introduces odd tokens (ex0: acronym).
- both models can truncate; tune generation length penalties and min/max lengths to reduce this (ex4).

## data + model
- dataset: [elrc‑medical‑v2 (en‑fr)](https://huggingface.co/datasets/qanastek/ELRC-Medical-V2)
- base model: `helsinki-nlp/opus-mt-en-fr`
- trainer: `transformers` seq2seq with early stopping on bleu

## reproduce
- install deps
```bash
pip install "accelerate>=0.26.0" transformers datasets evaluate sacrebleu tf-keras ipywidgets sentencepiece sacremoses nltk bert_score
```
- open and run `mt-mar3.ipynb` end‑to‑end (includes training, evaluation, and side‑by‑side samples).

## evaluation
- metrics: sacrebleu, meteor, bertscore (f1).
- split: 80/20 train/test from elrc‑medical‑v2.
- early stopping keyed to bleu; best checkpoint reported above.


