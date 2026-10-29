# Annotation guidelines: tweet sentiment (DRAFT, needs human review)

Version 0.1. Revise after the pilot (100 tweets, 2 to 3 annotators). Record every change at the bottom.

## The question you answer

**What sentiment does the author express in this tweet?** Choose one:

| Label | Use when the author expresses... |
|---|---|
| **positive** | approval, praise, happiness, hope, gratitude, support stated with positive emotion |
| **negative** | criticism, anger, fear, sadness, mockery, disappointment, contempt |
| **neutral** | no clear emotion: facts, news, questions, announcements, mixed feelings that cancel out |

Label the tweet as written. Do not use any tool, model score or search result while labelling, and do not
discuss items with other annotators until the round is finished.

## Rules

1. **Sentiment, not stance.** Judge the emotion expressed, not which party the author supports. "Modi won the
   election" is neutral. "So proud that Modi won!" is positive. "Modi won, God help us" is negative.
2. **Facts that are good or bad news are still neutral** unless the author adds feeling. "Unemployment rose to
   7%" is neutral; "Unemployment at 7%, shameful" is negative.
3. **Sarcasm: label the intended meaning** and tick `sarcasm`. "Great, another power cut. Thanks government"
   is negative.
4. **Questions** are neutral unless they are rhetorical attacks or praise ("Who else would vote for this
   clown?" is negative).
5. **Mixed sentiment:** if one side clearly dominates, label it; if they balance, choose neutral.
6. **Hindi / Hinglish / transliteration:** label what you understand and tick `code_mixed`. If you cannot
   understand enough to decide, choose your best guess and tick `unclear`.
7. **Slogans and hashtags alone** ("#ModiHaiToMumkinHai", "vote for X") are positive only if they cheer; a
   bare instruction to vote is neutral.
8. **Quoting someone else:** label the author's attitude to the quote if visible; otherwise neutral.
9. Tick `stance_differs` when the political stance you infer differs from the sentiment you chose (for
   example, a calm factual tweet that clearly supports a party).

## Worked examples (invented for illustration)

| Tweet | Label | Flags | Why |
|---|---|---|---|
| what a fantastic speech by the pm today proud of india | positive | | praise and pride |
| finally a government that delivers thank you | positive | | gratitude |
| hoping the new scheme helps farmers this year | positive | | hope |
| love how the rallies bring everyone together | positive | | approval |
| congratulations to all candidates who won | positive | | congratulations |
| modi hai toh mumkin hai | positive | code_mixed | cheering slogan |
| this government has ruined the economy | negative | | criticism |
| another scam another lie nothing changes | negative | | anger, contempt |
| scared of what happens to minorities after the results | negative | | fear |
| wow 15 lakh in every account still waiting | negative | sarcasm | mockery of an unmet promise |
| shameful silence on unemployment | negative | | criticism |
| kya bakwas hai yeh sarkar | negative | code_mixed | "what nonsense this government is" |
| polling for phase 3 starts tomorrow at 7am | neutral | | announcement |
| bjp won 303 seats in the lok sabha | neutral | stance_differs? | fact; tick the flag only if stance is evident |
| what time does the counting start | neutral | | question |
| pm to address the nation at 8pm | neutral | | news |
| good roads but bad jobs situation | neutral | | balanced mix |
| rahul gandhi visits kerala | neutral | | fact |

## Labelling tool and files

- Label Studio: import `data/gold/to_annotate.csv`, use `docs/label_studio_config.xml`, export CSV.
- Spreadsheet alternative: copy `to_annotate.csv`, add columns `label` (negative / neutral / positive) and
  `flags` (semicolon-separated: `sarcasm;code_mixed;stance_differs;unclear`). Save one file per annotator,
  e.g. `ann_a.csv`.
- Time guide: about 15 to 20 seconds per tweet; take a break every 150 tweets.

## Changelog

- 0.1: initial draft, before the pilot.
