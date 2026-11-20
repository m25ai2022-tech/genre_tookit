GENRE TOOLKIT
Interactive demo for "Interpretable Temporal Analysis of Fiction vs Non-fiction Text"

Gaurav Aggarwal (M25AI2022), M.Tech. (AI), IIT Jodhpur
Supervisor: Prof. Dr. Sidharth Ranjan


1. WHAT IT DOES
---------------

You paste a passage of text and the toolkit tells you whether it looks like fiction or
non-fiction, and shows why. It runs the same stages that are studied in the thesis, side by
side:

  1. Base paper (Qureshi 2019)           six document-level ratio features
  2. Short-text features (Kazmi 2022)    paragraph-level feature families
  3. + Handcrafted temporal features     the seven tense/aspect features added
  4. Neural network, no time             CNN + BiLSTM on GloVe vectors
  5. Narrative-time neural (Time2Vec)    the same network with a position-based time encoding

For every stage it prints the predicted class and P(fiction). For the feature-based stages it
also shows the tense of each sentence, the value of each temporal feature against the fiction
and non-fiction averages, and how much each feature pushed the decision.

Input needs at least 3 sentences. Longer text is cut into windows of 6 sentences, and each
window is analysed like a paragraph.

The accuracy figures shown in the app come from results.json (the thesis experiments:
file-grouped splits, five-seed averages). They are not the test scores of the single models
bundled here.


2. FOLDER CONTENTS
------------------

  app.py                  web server (Flask), http://127.0.0.1:5057
  index.html              the web page
  cli.py                  command-line version
  run_demo.sh             starts the server and opens the browser
  classify.py             prediction code (text in, verdicts and explanations out)
  features.py             feature extraction (Stanza parse, tense, POS ratios, etc.)
  basepaper.py            base-paper features
  neural.py               neural model definitions
  select_features.py      retraining step 1: feature selection (RFECV)
  train.py                retraining step 2: short-text and temporal-feature models
  train_basepaper.py      retraining step 3: base-paper model
  train_neural.py         retraining step 4: neural models
  results.json            accuracy figures displayed in the app
  notebook_features.json  feature lists taken from the evaluation notebooks
  notebook_cache/         cached feature-selection result used by select_features.py
  models/                 trained models used by the app
      toolkit.joblib                      short-text + temporal-feature models
      toolkit_allfeatures_backup.joblib   backup of the same models before feature pruning
      basepaper.joblib                    base-paper model
      neural.pt                           neural models (static and time-aware)
      brown_parses.pkl                    cached Stanza parses of the Brown paragraphs

The trained models are already included, so the app runs as it is. Retraining is optional.


3. REQUIREMENTS
---------------

  - Python 3.9 or newer
  - packages: numpy, scipy, pandas, scikit-learn, joblib, nltk, stanza, torch, flask
    (exact versions are in ../requirements.txt)
  - internet once, on the first run on a new machine: Stanza downloads its English model
    (tokenizer, tagger, lemmatiser and dependency parser). After that it works offline.
  - no GPU needed (the neural models run on the CPU)

Install:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r ../requirements.txt


4. RUN THE DEMO
---------------

From inside this folder:

    ./run_demo.sh

or

    python3 app.py

then open http://127.0.0.1:5057 in a browser. Start-up takes about 25 seconds while the models
and the parser load. The page has a text box, a "Load unseen Brown paragraph" button that fills the box
with a random paragraph from the held-out Brown files, and a results area with the five verdicts and the tense table.

If run_demo.sh says permission denied, run "chmod +x run_demo.sh" first, or use python3 app.py.

Stop the server with Ctrl+C.


5. COMMAND-LINE VERSION
-----------------------

If a browser is not available, pipe the text in:

    echo "She walked into the room. The lights were off. He had left hours ago. She sat down and cried. Nobody came." | python3 cli.py

Output is one line per stage with the predicted class and P(fiction), followed by a table for
each 6-sentence window with the tense of each sentence and the seven temporal feature values
next to the fiction and non-fiction averages.


6. RETRAIN THE MODELS (OPTIONAL)
--------------------------------

Run the four scripts in this order, from inside this folder. They overwrite the files in
models/, so copy that folder first if you want to keep the originals.

    python3 select_features.py     about 1 minute
    python3 train.py               about 1 minute
    python3 train_basepaper.py     about 10 minutes
    python3 train_neural.py        about 10 minutes on a CPU

The first two I re-ran on a laptop; the last two times are earlier estimates.

Before retraining, two paths must be in place.

  a) train.py reads ../brown_corpus/brown.csv, so a folder called brown_corpus must sit next to
     this folder, in Pipeline/. Unzip the Brown corpus from DataSet and link or copy it. From the
     Pipeline folder:

         cd DataSet
         unzip -q brown_corpus.zip -x "__MACOSX/*" "*.DS_Store"
         cd ..
         ln -s "$PWD/DataSet/brown_corpus" "$PWD/brown_corpus"

     Check that Pipeline/brown_corpus/brown.csv exists.

  b) train_neural.py reads the GloVe 100d vectors from ~/Downloads/glove.6B.100d.txt (line 16,
     GLOVE = ...). Extract that one file from DataSet/glove.6B.zip and copy it there, or edit
     the line to point to your copy:

         unzip -q -j DataSet/glove.6B.zip glove.6B.100d.txt -d ~/Downloads

Expected results after retraining (cross-validation on Brown paragraphs):
  short-text features (E1)        about 0.89 accuracy
  + temporal features (E3)        about 0.94 accuracy
On the demo's held-out split the single models score about 0.88 (E1) and 0.93 (E3).

These are single trained models on one split, so they differ a little from the five-seed
averages in the thesis and in results.json. Small differences between runs are normal.


7. TROUBLESHOOTING
------------------

  "Need at least 3 sentences"
      The text is too short. Paste a longer passage.

  ModuleNotFoundError (flask, stanza, torch, ...)
      Install the packages: pip install -r ../requirements.txt

  Stanza download fails or hangs on first start
      It needs internet once. Check the connection, then start the app again.

  Port 5057 already in use
      Close the other program, or change the port in the last line of app.py.

  FileNotFoundError: ../brown_corpus/brown.csv   (when retraining)
      Follow step 6a to create Pipeline/brown_corpus.

  FileNotFoundError: glove.6B.100d.txt   (when retraining the neural models)
      Follow step 6b.

  Page loads but nothing happens after pressing the button
      The first request can take a few seconds while the parser works. If the terminal shows an
      error, copy it and check the points above.

For the notebooks that produced the thesis results, see ../README.txt.
