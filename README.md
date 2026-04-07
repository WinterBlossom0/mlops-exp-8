# Experiment 8: CI Pipeline with GitHub Actions and CML

This experiment builds a simple machine learning CI/CD pipeline for a regression model. The project trains a model, logs metrics, generates a plot, serves predictions with Flask, and validates the API automatically through GitHub Actions on the `main` branch.

## Project files

- `model1.py`: trains the linear regression model, writes `metrics.txt`, and saves `model_results.png`
- `app.py`: exposes `/` and `/predict` endpoints using Flask
- `.github/workflows/main.yml`: automates install, train, report, deploy smoke test, and API check
- `requirements.txt`: Python dependencies for local runs and CI

## How to run locally

```bash
pip install -r requirements.txt
python model1.py
python app.py
```

Open `http://127.0.0.1:5000/`, then test the API from another terminal:

```bash
curl -X POST "http://127.0.0.1:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"input":[5]}'
```

## Workflow summary

The GitHub Actions workflow in `.github/workflows/main.yml` runs whenever code is pushed to `main`.

1. It checks out the repository.
2. It installs Python and the project dependencies.
3. It installs CML.
4. It trains the model by running `model1.py`.
5. It creates a markdown report using `metrics.txt` and `model_results.png`.
6. It publishes the report back to GitHub using CML.
7. It starts the Flask API.
8. It sends a `curl` request to `/predict` to confirm deployment is working.

## Post-lab answers

### 1. What is `main.yml`, and explain the workflow of it.

`main.yml` is the GitHub Actions workflow file stored in `.github/workflows/`. It defines the automation steps that GitHub runs for the repository. In this experiment, `main.yml` is responsible for continuous integration and a small continuous deployment check:

- `on`: triggers the workflow when code is pushed to `main`
- `permissions`: grants the workflow token enough access to publish the CML report
- `jobs`: defines the `train-test-deploy` job
- `steps`: installs dependencies, trains the model, prepares the report, starts Flask, and tests the API

In short, `main.yml` is the file that turns normal code pushes into an automated MLOps pipeline.

### 2. How does CML work? Explain with a diagram.

CML connects machine learning scripts with CI platforms such as GitHub Actions. After the workflow runs, it collects generated outputs like metrics, plots, and evaluation summaries, then posts them back to GitHub as comments or reports.

```text
Developer push to main
          |
          v
GitHub Actions workflow starts
          |
          v
Install dependencies + CML
          |
          v
Run training script (model1.py)
          |
          v
Generate metrics.txt + model_results.png
          |
          v
Create report.md
          |
          v
CML posts results back to GitHub
          |
          v
Start Flask API and test /predict
```

This makes model training, validation, and result sharing reproducible and automatic.

## Notes

The lab manual shows older CML commands such as `cml-publish` and `cml-send-comment`. This implementation uses the current CML GitHub Action setup and `cml comment create`, while keeping the same experiment outcome.

## References

- GitHub Actions workflow syntax: https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions
- CML with GitHub Actions: https://cml.dev/doc/start/github
- CML GitHub integration: https://cml.dev/doc/ref/comment
- Setup CML action: https://github.com/iterative/setup-cml
