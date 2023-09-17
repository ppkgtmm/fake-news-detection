# Data visualization

## Label distribution
<img width=500 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/label_distribution.jpg" />

**Observation**
- Proportion of fake news is 52 % which is higher than real news by 4 %
- Taking the proportion of class labels into accout, we are unlikely to face class imbalance problem

## Subject distribution
- REAL news
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/real_subject_distribution.jpg" />

- FAKE news
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/fake_subject_distribution.jpg" />

**Observation**
- Real news are well organized into 2 categories namely politics news and world news
- On the other hand, fakes news subjects are more spread out for example, government can be merged into politics subject, U.S. and Middle-east news 
can be merge to world news

## Wordcloud
- REAL news
<img width=900 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/real_text_word_cloud.png" />

- FAKE news
<img width=900 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/fake_text_word_cloud.png" />

**Observation**
- Both real and fake news contain words related to politics, U.S. politicians, social issues and crime. Also, there are occurrences of country names, numbers, emails, mentions and hashtags in both type of news
- Unlike the real news, fake news collected contains informal words, urls, social media website names

## News word count distribution
- REAL news
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/real_text_word_count_dist.jpg" />

- FAKE news
<img width=700  src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/fake_text_word_count_dist.jpg" />

**Observation**
- Word count of each news, obtained by counting no. of parts after splitting news by space, was used to plot the above histograms
- Looking at the overall trend, both real news and fake news word count distribution plot are right skewed which means there exist some news that are longer than majority
- Most real news contains not more than a thousand words whereas serveral fake news in the data collected tend to be longer in terms of word count

## News average word length distribution
- REAL news
  
<p align="center">
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/real_text_avg_word_len_dist.jpg" />
</p>

- FAKE news
  
<p align="center">
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/fake_text_avg_word_len_dist.jpg" />
</p>

**Observation**
- Average word length of each news, obtained by counting no. of characters in each word inside the news and averaging the count, was used to plot the above histograms
- Interestingly, real news average word length distribution plot looks like symmetric bell-shaped curve i.e. normal distribution
- Fake news average word length distribution is also symmetric but less due to outlier data points(s) on the left hand side of curve. Furthermore, range of average word length in fake news is wider than real news
