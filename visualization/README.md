# Data visualization

### Label distribution
<img width=600 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/label_distribution.jpg" />

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
- On the other hand, fakes news subjects are more spread out for exmaple, government can be merged into politics subject, U.S. and Middle-east news 
can be merge to world news

## Wordcloud
- REAL news
<img width=900 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/real_text_word_cloud.png" />

- FAKE news
<img width=900 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/fake_text_word_cloud.png" />

**Observation**
- Both real and fake news contain words related to politics, U.S. politicians, social issues and crime. Also, there are occurrences of country names, numbers, emails, mentions and hashtags in both type of news
- Unlike the real news, fake news collected contains informal words, urls, social media platform and website names

## News word count distribution
- REAL news
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/real_text_word_count_dist.jpg" />

- FAKE news
<img width=700 src="https://github.com/ppkgtmm/fake-news-detection/raw/main/visualization/outputs/fake_text_word_count_dist.jpg" />

**Observation**
- Looking at the overall trend, real news word count distribution plot shape is j-shaped. Moreover, most real news contains not more than a thousand words
- Fake news word count distribution is right skewed which means there are some fake news that are extremely longer than majority. Apart from that, many fake news in the data collected are longer than real news
