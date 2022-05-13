# TestExercise
Topic Trending Natural Language Processing
Test exercise
Goal 1: Extract topics from docs (define topics however you want)
Goal 2: Find trends in topics (trends can be anything you want)

The code is made up of 12 steps which was the main purpose of creating an algorithm for theme trends.
1. The Packages used for different processing inside code.
Add requirements.txt like list of all projects dependencies.
Install requirements using this function:
pip freeze > requirements.txt
pip install -r requirements.txt
3. Input from user. That is data which we will use in process. Data can be in different format. In our case, that is csv file.
4. Data cleaning is very important part. We filtering data removing emails, quotes and other distractions. 
5. Purpose of stemming is reducing a word afixes sufixes, prefixes or roots.
6. For LatentDirichletAllocation(LDA) necessary is that we have Document-Word Matrix.
7. Building LDA Topic Model. This is not only one mode for NLP, but this is most useful.
8. Creating Topic names based on LDA model
9. Last 4 steps, we were foucusing on creation keyword topic, dominant topic and show that in JSON format
Furthermore, this solution we can improve with GRID searching best LDA model and apply that in process. 
In conclusion, we can say that this process is very interesting, but require good background in mathematics and statistics.
