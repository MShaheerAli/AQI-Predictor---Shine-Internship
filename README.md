# Pearls AQI Predictor — Lahore

Everything is already written for you. This file is just the sequence of
clicks and commands to get it running. Follow it top to bottom, in order.
Don't skip ahead.

---

## PART A — Accounts (do this first, ~20 min)

1. **OpenWeather account** → go to openweathermap.org → sign up (free) →
   go to your account → "API keys" tab → copy the key shown there.
   (New keys can take up to ~1 hour to activate, so do this step first.)

2. **Hopsworks account** → go to app.hopsworks.ai → sign up with Google
   or GitHub (free) → create a new project (any name, e.g. "aqi") →
   inside the project, go to Account Settings → API Keys → generate a
   new key → copy it. Keep this tab open, you'll need it again.

3. **GitHub account + repo** → if you don't have GitHub, sign up at
   github.com → click "New repository" → name it `aqi-predictor` →
   keep it Public → create it (don't add a README, we already have one).

---

## PART B — Get the code running on your own laptop (~30 min)

4. **Install Python** → go to python.org/downloads → download and
   install the latest version → during install, tick the box that says
   "Add Python to PATH".

5. **Open a terminal** in this project folder (on Windows: right-click
   inside the folder → "Open in Terminal"; on Mac: right-click → "New
   Terminal at Folder").

6. **Create your keys file** → in this folder, find `.env.example` →
   make a copy of it and rename the copy to exactly `.env` → open it in
   any text editor → paste your OpenWeather key and Hopsworks key in
   (replacing the placeholder text) → save.

7. **Install the required packages** → in the terminal, type:
   ```
   pip install -r requirements.txt
   ```
   Wait for it to finish (a few minutes).

---

## PART C — Build your data + model (~2-3 hrs, mostly waiting)

Run these commands **one at a time**, in this exact order, waiting for
each to finish and print "Done"/no errors before running the next:

8. ```
   python feature_pipeline.py
   ```
   This grabs today's live Lahore weather + pollution and stores it.
   If this fails, the error message will tell you if it's your
   OpenWeather key or Hopsworks key that's wrong — fix your `.env` file
   and try again.

9. ```
   python backfill_historical.py
   ```
   This pulls ~45 days of past data so you have enough to train a
   model on. Takes a few minutes.

10. ```
    python training_pipeline.py
    ```
    This trains 3 different models, prints their accuracy scores, and
    saves the best one. You'll see RMSE/MAE/R² numbers printed — save
    a screenshot or copy them, you need them for your report.

---

## PART D — See your dashboard locally (~5 min)

11. ```
    streamlit run app.py
    ```
    A browser tab will open automatically showing your live dashboard.
    Check that it shows a forecast chart and a metrics table. Leave
    this running while you look at it; press Ctrl+C in the terminal to
    stop it when you're done.

---

## PART E — Put it on the internet (~30 min)

12. **Push your code to GitHub**:
    ```
    git init
    git add .
    git commit -m "AQI predictor"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/aqi-predictor.git
    git push -u origin main
    ```
    (Replace YOUR_USERNAME with your actual GitHub username. If `git`
    isn't installed, download it from git-scm.com first.)

13. **Add your keys as GitHub Secrets** (so the automation can run
    without your `.env` file, which never gets uploaded):
    On your repo's GitHub page → Settings → Secrets and variables →
    Actions → "New repository secret" → add one called
    `OPENWEATHER_API_KEY` with your key → add another called
    `HOPSWORKS_API_KEY` with your key.

14. **Check the automation is working**: on your repo's GitHub page →
    "Actions" tab → you should see "Hourly Feature Pipeline" and
    "Daily Training Pipeline" listed. Click into one → "Run workflow"
    to trigger it manually right now and confirm it succeeds (green
    checkmark). This is what satisfies the CI/CD requirement.

15. **Deploy the dashboard live**: go to share.streamlit.io → sign in
    with GitHub → "New app" → pick your `aqi-predictor` repo → main
    file path: `app.py` → in "Advanced settings", add your two secrets
    the same way (`OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`) → Deploy.
    You'll get a public URL — this is your live dashboard link to submit.

---

## PART F — Write and submit the report (~1.5-2 hrs)

16. Open `REPORT_TEMPLATE.md` in this folder — it's a fill-in-the-blank
    report structure already matching what your grader asked for.
    Fill in the metrics table from step 10, add screenshots of your
    dashboard (step 11 or 15) and your GitHub Actions runs (step 14),
    and answer the short-answer sections in your own words.

17. **Submit**: your GitHub repo link, your live Streamlit URL, and the
    filled-in report (convert `REPORT_TEMPLATE.md` to PDF/Word, or just
    submit it as-is if Markdown is accepted).

---

## If something breaks

- **"Invalid API key" error from OpenWeather**: new keys take up to an
  hour to activate. Wait and retry.
- **Hopsworks login fails**: double check you copied the whole API key
  with no extra spaces into `.env`.
- **`training_pipeline.py` says "only X usable rows"**: your backfill
  didn't get enough data — re-run `backfill_historical.py 60` (60 days
  instead of the default 45) or wait longer before training.
- **Streamlit Cloud deploy fails on `tensorflow` or a heavy package**:
  this project intentionally uses lightweight scikit-learn models
  (Ridge / Random Forest / a small neural net) instead of TensorFlow so
  this doesn't happen — if you added TensorFlow yourself, remove it.
"# AQI-Predictor---Shine-Internship" 
