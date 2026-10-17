# Cleaner choice challenge

A fictional US brand wants to sell a new toilet cleaner. The bottle holds 16 fl oz. Its formula stays the same. The brand can change its price and main shelf message. Each cleaner makes different claims. The data do not prove those claims.

The public data show six visits for each of 250 fictional households. A household could buy one cleaner or buy none. Some households gave short answers after a choice or after using a cleaner. Their answers can be incomplete or wrong. The starter model gives a chance for each choice and a short reason. The data cannot prove why a household chose an item. They cannot show how real shoppers will act.

**Your task:** The brand can fund one US test of the new cleaner. Recommend a price, shelf message, and sales channel to test first. If the evidence is too weak, say "I do not know yet." Explain why. Name the data or test that would let you decide. You have 48 hours to submit. A careful, partial result is useful.

## 1. Build your approach

Before you change code, commit a file named `FIRST.md`. Use at most 120 words. State your first concern, your first test, and what result would change your mind. Work in as many stages as you find useful.

Run the test. Change how the system makes or explains choices across visits. Show household-level reasons and evidence for trusting or doubting them. You may change any part of the model or data. You may replace the approach. If you add data, state its source. Explain your choices. State what the system carries from one visit to the next, if anything.

## 2. Check trust

Build checks that help you decide whether the brand can trust the system. Choose what to check and how. Show an idea that failed, gave conflicting results, or remains untested. Explain why. State what the data cannot tell you. Describe a test with real US shoppers that could show the system is wrong. State what result would change your view. You do not need to run this test.

## Show your work

- Give us a live app or endpoint. It must accept a changed input and return a new result from your code. A simple interface is enough.
- Before each experiment, record what you expect to see. Save the expectation, inputs, outputs, model version, and data version in a database. Keep your first run and failed runs. Let us inspect past runs by ID. You may use SQLite or any free database.
- Send your repository link, live link, one command to run your work locally, and a copy or export of the run database.
- Add a final note of at most 200 words. Name a saved run that changed or confirmed your view. Choose a next test for the brand. If no test is ready, say why. State what result would change your choice.

You may use AI tools. Your program should run on a laptop CPU. It should not need an API or GPU. We look at how you frame the problem, test ideas, change course, and state limits. We care more about your decisions and evidence than a score or a polished interface.

We will ask you to explain one saved run and a new household in a short live review.

## Start

Use Python 3.10 or later. Run `python run_baseline.py` from this folder. The starter uses the Python standard library. It prints results for a build set and a check set.

`households.csv` holds answers from one survey before the visits. It includes income bands, budget pressure, and reported monthly spend on cleaning supplies.

`products.csv` describes the four cleaners. `visits.csv` holds 1,200 visits from 200 households. Use it to build your approach. `check_visits.csv` holds 300 visits from 50 other households. Use it to test your approach. If its results lead you to change your approach, say so. Do not present a later result on that same set as a fresh test. Both visit files hold the channel, prices, message, and choice. They also hold some answers given after the choice (`said_why`) and after product use (`after_use_note`). These answers are available before the next visit. Visits appear in time order for each household. Some fields are blank. Prices are in US dollars. All households, prices, claims, and answers are fictional.
