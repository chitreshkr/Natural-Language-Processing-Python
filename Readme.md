## Complete Text Processing 


```python
import pandas as pd
import numpy as np
import spacy
```


```python
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
```


```python
df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv', encoding = 'latin1')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>is bored and wants to watch a movie  any sugge...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>back in miami.  waiting to unboard ship</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@misskpey awwww dnt dis brng bak memoriessss, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ughhh i am so tired  blahhhhhhhhh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@mandagoforth me bad! It's funny though. Zacha...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>i just graduated</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>Templating works; it all has to be done</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>mommy just brought me starbucks</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>@omarepps watching you on a House re-run...lov...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>Thanks for trying to make me smile I'll make y...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 2 columns</p>
</div>




```python
df['sentiment'].value_counts()
```




    1    2000
    0    2000
    Name: sentiment, dtype: int64



## Word Counts


```python
len('this is text'.split())
```




    3




```python
df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2296</th>
      <td>bulat dan bahagia  and desperately needing a k...</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>@johncmayer Like there was ever any doubt you ...</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2468</th>
      <td>@kirstiealley  LETS DO IT!</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>66</th>
      <td>@anthothemantho hahaha i agree! i cried like a...</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1602</th>
      <td>@KINOFLYHIGH fuck i shouldnt have left!</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['word_counts'].max()
```




    32




```python
df['word_counts'].min()
```




    1




```python
df[df['word_counts']==1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>385</th>
      <td>homework</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>691</th>
      <td>@ekrelly</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1124</th>
      <td>disappointed</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1286</th>
      <td>@officialmgnfox</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>headache</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>@MCRmuffin</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2542</th>
      <td>Graduated!</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2947</th>
      <td>reading</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3176</th>
      <td>@omeirdeleon</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>www.myspace.com/myfinalthought</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3966</th>
      <td>@gethyp3</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Characters Count


```python
len('this is')
```




    7




```python
def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)
```


```python
char_counts('this is')
```




    6




```python
df['char_counts'] = df['twitts'].apply(lambda x: char_counts(str(x)))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2503</th>
      <td>Woke up. Such a nice weather out there. Shower...</td>
      <td>1</td>
      <td>13</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1408</th>
      <td>I think I killed outlook</td>
      <td>0</td>
      <td>5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>16</th>
      <td>@BrianQuest I made 1 fo u 2: http://bit.ly/eId...</td>
      <td>0</td>
      <td>19</td>
      <td>81</td>
    </tr>
    <tr>
      <th>601</th>
      <td>working all day on mothers dayy  but i left my...</td>
      <td>0</td>
      <td>20</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>This java assignment has really got me down.  ...</td>
      <td>0</td>
      <td>24</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
</div>



## Average Word Length


```python
x = 'this is' # 6/2 = 3
y = 'thankyou guys' # 12/2 = 6
```


```python
df['avg_word_len'] = df['char_counts']/df['word_counts']
```


```python
df.sample(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>489</th>
      <td>thiking of goin to the library but not realy c...</td>
      <td>0</td>
      <td>11</td>
      <td>52</td>
      <td>4.727273</td>
    </tr>
    <tr>
      <th>1291</th>
      <td>I dropped one of my iPod earphones in a glass ...</td>
      <td>0</td>
      <td>12</td>
      <td>43</td>
      <td>3.583333</td>
    </tr>
    <tr>
      <th>1834</th>
      <td>carley &amp;amp; kim are coming over! but no mallo...</td>
      <td>0</td>
      <td>17</td>
      <td>71</td>
      <td>4.176471</td>
    </tr>
    <tr>
      <th>1494</th>
      <td>I'm still alive, but I need some miracle. Don'...</td>
      <td>0</td>
      <td>23</td>
      <td>91</td>
      <td>3.956522</td>
    </tr>
  </tbody>
</table>
</div>



## Stop Words Count 


```python
print(stopwords)
```

    {'go', 'alone', 'besides', 'against', 'anyway', 'being', 'former', 'becoming', 'namely', 'this', 'over', 'whole', "'s", 'name', 'were', 'nevertheless', 'herein', 'nowhere', 'more', 'whether', 'amount', 'per', 'everything', 'our', 'than', 'show', 'top', 'them', '’s', 'how', 'on', 'my', 'mostly', 'done', 'seems', 'serious', 'both', 'very', 'amongst', 'who', 'n‘t', 'often', 'twenty', 'thus', '’ve', 'should', 'few', 'again', 'hundred', 'any', 'under', 'become', 'three', 'must', 'twelve', 're', 'meanwhile', 'also', 'around', 'out', 'something', 'other', 'whither', 'after', 'these', 'using', 'else', 'further', 'see', 'down', 'side', 'each', 'one', 'cannot', 'within', 'us', 'whereas', "'m", 'somehow', 'elsewhere', 'its', 'but', 'seemed', 'made', 'hers', '‘s', 'the', '’m', 'at', 'his', "'ve", 'another', 'perhaps', 'became', 'those', 'least', 'nine', 'she', '‘ll', '‘m', 'it', 'are', 'either', 'not', 'ten', '’re', 'you', 'has', 'still', 'off', 'sometimes', 'is', 'had', 'whom', 'why', 'with', 'used', 'say', 'could', 'was', 'yours', 'therein', 'when', 'enough', 'rather', 'yourselves', 'throughout', 'her', 'because', 'seem', 'fifteen', 'in', 'keep', 'just', 'fifty', 'quite', '’d', 'five', 'across', 'then', 'their', 'therefore', 'already', 'moreover', 'up', '‘d', 'have', 'put', 'that', 'there', 'onto', 'herself', 'most', 'no', 'whatever', 'since', 'though', 'may', 'ca', 'from', 'someone', 'latter', 'eight', 'they', 'and', 'various', 'well', 'latterly', 'whereafter', 'now', 'anything', 'ourselves', "'re", 'into', "n't", 'somewhere', 'an', 'take', 'been', 'without', 'indeed', 'me', 'third', 'thru', 'him', 'whereupon', 'whoever', 'above', 'next', 'which', 'themselves', 'several', 'last', 'four', 'many', 'thence', 'whereby', 'beyond', 'between', 'much', 'however', 'seeming', 'hereby', 'unless', 'hence', 'n’t', 'yet', 'nor', '‘ve', 'along', 'although', 'among', 'via', 'never', 'give', 'regarding', 'wherever', 'to', 'he', 'would', 'of', 'mine', 'always', 'back', 'anyone', 'others', 'do', 'two', 'until', 'your', 'as', 'bottom', 'thereafter', 'formerly', 'neither', 'toward', 'we', 'thereupon', 'all', 'together', 'becomes', '‘re', 'so', 'might', 'thereby', 'empty', 'where', 'please', 'ours', 'will', 'move', "'ll", 'even', 'or', 'myself', 'afterwards', 'does', 'front', 'get', 'anywhere', 'nothing', 'own', 'am', 'beforehand', 'behind', 'by', 'too', 'doing', 'beside', 'wherein', 'i', 'be', 'whose', 'if', 'such', 'did', 'less', 'otherwise', 'part', 'make', 'noone', 'every', 'due', 'almost', 'except', 'before', 'what', 'some', 'same', 'ever', 'everyone', 'here', 'while', 'a', 'hereupon', 'about', 'none', 'call', '’ll', 'whence', 'eleven', 'anyhow', 'hereafter', 'for', 'itself', 'once', 'six', 'nobody', 'sixty', 'only', 'first', 'really', 'towards', 'whenever', 'yourself', 'himself', 'below', 'everywhere', 'forty', 'upon', 'through', 'full', "'d", 'sometime', 'can', 'during'}
    


```python
len(stopwords)
```




    326




```python
x = 'this is the text data'
```


```python
x.split()
```




    ['this', 'is', 'the', 'text', 'data']




```python
[t for t in x.split() if t in stopwords]
```




    ['this', 'is', 'the']




```python
len([t for t in x.split() if t in stopwords])
```




    3




```python
df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in stopwords]))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1692</th>
      <td>@Person678 Keep trying, I grew one last year a...</td>
      <td>0</td>
      <td>23</td>
      <td>90</td>
      <td>3.913043</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3021</th>
      <td>@taylormcfly I know!! Should of guessed they'd...</td>
      <td>1</td>
      <td>10</td>
      <td>51</td>
      <td>5.100000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1544</th>
      <td>Although I want to hit up mcdonalds breakfast ...</td>
      <td>0</td>
      <td>9</td>
      <td>46</td>
      <td>5.111111</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1329</th>
      <td>i'm gonna be a good girl and stay at my dorm d...</td>
      <td>0</td>
      <td>23</td>
      <td>88</td>
      <td>3.826087</td>
      <td>13</td>
    </tr>
    <tr>
      <th>876</th>
      <td>@LipstickNYC hmmm i owed you a story yesterday...</td>
      <td>0</td>
      <td>23</td>
      <td>114</td>
      <td>4.956522</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```

## Count #HashTags and @Mentions 


```python
x = 'this is #hashtag and this is @mention'
```


```python
x.split()
```




    ['this', 'is', '#hashtag', 'and', 'this', 'is', '@mention']




```python
[t for t in x.split() if t.startswith('@')]
```




    ['@mention']




```python
len([t for t in x.split() if t.startswith('@')])
```




    1




```python
df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))
```


```python
df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>843</th>
      <td>@Ms_Kaydine all im sayin is MJ's feet better g...</td>
      <td>0</td>
      <td>26</td>
      <td>109</td>
      <td>4.192308</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2597</th>
      <td>angels and demonds...i saw that movie yesterda...</td>
      <td>1</td>
      <td>10</td>
      <td>67</td>
      <td>6.700000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>657</th>
      <td>I need a bf! LOL anyone wanna sign up haha. Th...</td>
      <td>0</td>
      <td>32</td>
      <td>105</td>
      <td>3.281250</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>@ABBSound ??????? ????? ??? ???? ??? ??? ?? ??...</td>
      <td>0</td>
      <td>9</td>
      <td>46</td>
      <td>5.111111</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2335</th>
      <td>@lukeb3000 i might be interested. how shall i ...</td>
      <td>1</td>
      <td>10</td>
      <td>52</td>
      <td>5.200000</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## If numeric digits are present in twitts


```python
x = 'this is 1 and 2'
```


```python
x.split()
```




    ['this', 'is', '1', 'and', '2']




```python
x.split()[3].isdigit()
```




    False




```python
[t for t in x.split() if t.isdigit()]
```




    ['1', '2']




```python
df['numerics_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1063</th>
      <td>@Destini41 where do you think the otalia story...</td>
      <td>0</td>
      <td>19</td>
      <td>95</td>
      <td>5.000000</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>546</th>
      <td>fml :/ today is too nice of a day to feel this...</td>
      <td>0</td>
      <td>13</td>
      <td>38</td>
      <td>2.923077</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3325</th>
      <td>@Kimmy6313 I totally feel better, you were rig...</td>
      <td>1</td>
      <td>14</td>
      <td>62</td>
      <td>4.428571</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>686</th>
      <td>wants tomorrow to be over already.</td>
      <td>0</td>
      <td>6</td>
      <td>29</td>
      <td>4.833333</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1814</th>
      <td>@xMarshmellows Awww</td>
      <td>0</td>
      <td>2</td>
      <td>18</td>
      <td>9.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## UPPER case words count 


```python
x = 'I AM HAPPY'
y = 'i am happy'
```


```python
[t for t in x.split() if t.isupper()]
```




    ['I', 'AM', 'HAPPY']




```python
df['upper_counts'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper()]))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1617</th>
      <td>thinks working 57 hours this week might just k...</td>
      <td>0</td>
      <td>22</td>
      <td>84</td>
      <td>3.818182</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>565</th>
      <td>@derrickkendall that is if i'm not busy murder...</td>
      <td>0</td>
      <td>17</td>
      <td>84</td>
      <td>4.941176</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>946</th>
      <td>Muwahahaha .... &amp;gt;.&amp;gt; hides behind pink fo...</td>
      <td>0</td>
      <td>21</td>
      <td>100</td>
      <td>4.761905</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1517</th>
      <td>Making pesto pasta for memy 2nd bday dinner! H...</td>
      <td>0</td>
      <td>28</td>
      <td>111</td>
      <td>3.964286</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3864</th>
      <td>The first day of &amp;quot;real&amp;quot; rehersals of...</td>
      <td>1</td>
      <td>21</td>
      <td>91</td>
      <td>4.333333</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[3962]['twitts']
```




    '@DavidArchie Our local shows love tributes too much. True story! Will be watching SIS videos in Youtube later, haha '




```python

```


```python

```

# Preprocessing and Cleaning

## Lower Case Conversion 


```python
x = 'this is Text'
```


```python
x.lower()
```




    'this is text'




```python
x = 45.0
str(x).lower()
```




    '45.0'




```python
df['twitts'] = df['twitts'].apply(lambda x: str(x).lower())
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1048</th>
      <td>afternoon everyone just playing some tunes whi...</td>
      <td>0</td>
      <td>24</td>
      <td>106</td>
      <td>4.416667</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>shit you mister gembul! oh no.. you stole my h...</td>
      <td>0</td>
      <td>10</td>
      <td>45</td>
      <td>4.500000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3709</th>
      <td>@silverlines hey you opened it!  congrats!</td>
      <td>1</td>
      <td>6</td>
      <td>36</td>
      <td>6.000000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1777</th>
      <td>@i140  myliferecord ... a health/medical histo...</td>
      <td>0</td>
      <td>16</td>
      <td>90</td>
      <td>5.625000</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>@talentdmrripley  maybe a good night's sleep f...</td>
      <td>0</td>
      <td>8</td>
      <td>50</td>
      <td>6.250000</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Contraction to Expansion 


```python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and ",
"won't": "would not",
'dis': 'this',
'bak': 'back',
'brng': 'bring'}
```


```python
x = "i'm don't he'll" # "i am do not he will"
```


```python
def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
    
```


```python
cont_to_exp(x)
```




    'i am do not he will'




```python
%%timeit
df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))
```

    97.6 ms ± 4.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3348</th>
      <td>@timtech awww, how cute. i love when men go al...</td>
      <td>1</td>
      <td>11</td>
      <td>47</td>
      <td>4.272727</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>470</th>
      <td>wii says i gained back .4 pounds</td>
      <td>0</td>
      <td>7</td>
      <td>26</td>
      <td>3.714286</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>826</th>
      <td>@littleliverbird maybe. i go on a bit less too...</td>
      <td>0</td>
      <td>27</td>
      <td>109</td>
      <td>4.037037</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>570</th>
      <td>cannot get into mariah's new song.</td>
      <td>0</td>
      <td>6</td>
      <td>28</td>
      <td>4.666667</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2966</th>
      <td>@sassyback dude i am gen y myself</td>
      <td>1</td>
      <td>6</td>
      <td>27</td>
      <td>4.500000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Count and Remove Emails 


```python
import re
```


```python
df[df['twitts'].str.contains('hotmail.com')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3713</th>
      <td>@securerecs arghh me please  markbradbury_16@h...</td>
      <td>1</td>
      <td>5</td>
      <td>51</td>
      <td>10.2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[3713]['twitts']
```




    '@securerecs arghh me please  markbradbury_16@hotmail.com'




```python
x = '@securerecs arghh me please  markbradbury_16@hotmail.com'
```


```python
re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x)
```




    ['markbradbury_16@hotmail.com']




```python
df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x))
```


```python
df['emails_count'] = df['emails'].apply(lambda x: len(x))
```


```python
df[df['emails_count']>0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3713</th>
      <td>@securerecs arghh me please  markbradbury_16@h...</td>
      <td>1</td>
      <td>5</td>
      <td>51</td>
      <td>10.2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[markbradbury_16@hotmail.com]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)
```




    '@securerecs arghh me please  '




```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x))
```


```python
df[df['emails_count']>0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3713</th>
      <td>@securerecs arghh me please</td>
      <td>1</td>
      <td>5</td>
      <td>51</td>
      <td>10.2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[markbradbury_16@hotmail.com]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Count URLs and Remove it 


```python
x = 'hi, thanks to watching it. for more visit https://youtube.com/kgptalkie'
```


```python
#shh://git@git.com:username/repo.git=riif?%
```


```python
re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
```




    [('https', 'youtube.com', '/kgptalkie')]




```python
df['url_flags'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))
```


```python
df[df['url_flags']>0].sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
      <th>url_flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3203</th>
      <td>@thewebguy http://twitpic.com/6jb33 - dude, th...</td>
      <td>1</td>
      <td>14</td>
      <td>85</td>
      <td>6.071429</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3362</th>
      <td>shabtai it is  great prizes today! (go almost ...</td>
      <td>1</td>
      <td>15</td>
      <td>80</td>
      <td>5.333333</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2537</th>
      <td>@seuj sardinia for a few days of pre-graduatio...</td>
      <td>1</td>
      <td>10</td>
      <td>67</td>
      <td>6.700000</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2458</th>
      <td>and again http://twitpic.com/4wp8l</td>
      <td>1</td>
      <td>3</td>
      <td>32</td>
      <td>10.666667</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>548</th>
      <td>@cyphersushi  no, i am afraid not.but! go here...</td>
      <td>0</td>
      <td>16</td>
      <td>117</td>
      <td>7.312500</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
x
```




    'hi, thanks to watching it. for more visit https://youtube.com/kgptalkie'




```python
re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)
```




    'hi, thanks to watching it. for more visit '




```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
      <th>url_flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2784</th>
      <td>@realadulttalk come on and smile for me? that ...</td>
      <td>1</td>
      <td>12</td>
      <td>62</td>
      <td>5.166667</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>@richmiller oh man, i am really sorry  i hope ...</td>
      <td>0</td>
      <td>17</td>
      <td>67</td>
      <td>3.941176</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>190</th>
      <td>im veryy bad</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>3.333333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>@simplymallory you be naht online d:  sighs  i...</td>
      <td>0</td>
      <td>15</td>
      <td>63</td>
      <td>4.200000</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>just got sad, although sadly expected, news fr...</td>
      <td>0</td>
      <td>10</td>
      <td>48</td>
      <td>4.800000</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Remove RT 


```python
df[df['twitts'].str.contains('rt')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
      <th>url_flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>@mandagoforth me bad! it is funny though. zach...</td>
      <td>0</td>
      <td>26</td>
      <td>116</td>
      <td>4.461538</td>
      <td>13</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ut oh, i wonder if the ram on the desktop is s...</td>
      <td>0</td>
      <td>14</td>
      <td>46</td>
      <td>3.285714</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>@paulmccourt dunno what sky you're looking at!...</td>
      <td>0</td>
      <td>15</td>
      <td>80</td>
      <td>5.333333</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>im back home in belfast  im realli tired thoug...</td>
      <td>0</td>
      <td>22</td>
      <td>84</td>
      <td>3.818182</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>@lilmonkee987 i know what you mean... i feel s...</td>
      <td>0</td>
      <td>11</td>
      <td>48</td>
      <td>4.363636</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3913</th>
      <td>for the press so after she recovered she kille...</td>
      <td>1</td>
      <td>24</td>
      <td>100</td>
      <td>4.166667</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3919</th>
      <td>earned her cpr &amp;amp; first aid certifications!</td>
      <td>1</td>
      <td>7</td>
      <td>40</td>
      <td>5.714286</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3945</th>
      <td>@teciav &amp;quot;i look high, i look low, i look ...</td>
      <td>1</td>
      <td>23</td>
      <td>106</td>
      <td>4.608696</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3951</th>
      <td>i am soo very parched. and hungry. oh and i am...</td>
      <td>1</td>
      <td>21</td>
      <td>87</td>
      <td>4.142857</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3986</th>
      <td>@countroshculla yeah..needed to get up early.....</td>
      <td>1</td>
      <td>10</td>
      <td>69</td>
      <td>6.900000</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>381 rows × 13 columns</p>
</div>




```python
x = 'rt @username: hello hirt'
```


```python
re.sub(r'\brt\b', '', x).strip()
```




    '@username: hello hirt'




```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'\brt\b', '', x).strip())
```


```python

```


```python

```


```python

```

## Special Chars removal or punctuation removal 


```python
df.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
      <th>url_flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2205</th>
      <td>eating food  leaving school to go to hospital ...</td>
      <td>1</td>
      <td>12</td>
      <td>45</td>
      <td>3.750000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>812</th>
      <td>@earthlifeshop i know! it makes it hard for th...</td>
      <td>0</td>
      <td>17</td>
      <td>74</td>
      <td>4.352941</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>cannot sleep! only 3 hours til i have to wake up</td>
      <td>0</td>
      <td>11</td>
      <td>38</td>
      <td>3.454545</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = '@duyku apparently i was not ready enough... i...'
```


```python
re.sub(r'[^\w ]+', "", x)
```




    'duyku apparently i was not ready enough i'




```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'[^\w ]+', "", x))
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitts</th>
      <th>sentiment</th>
      <th>word_counts</th>
      <th>char_counts</th>
      <th>avg_word_len</th>
      <th>stop_words_len</th>
      <th>hashtags_count</th>
      <th>mentions_count</th>
      <th>numerics_count</th>
      <th>upper_counts</th>
      <th>emails</th>
      <th>emails_count</th>
      <th>url_flags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2294</th>
      <td>joshishollywood aw joshi would describe you ex...</td>
      <td>1</td>
      <td>9</td>
      <td>55</td>
      <td>6.111111</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3495</th>
      <td>repressd i hate it when that happens errrr i m...</td>
      <td>1</td>
      <td>14</td>
      <td>63</td>
      <td>4.500000</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>but when you do have a camera less funny thing...</td>
      <td>0</td>
      <td>11</td>
      <td>45</td>
      <td>4.090909</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3702</th>
      <td>uh do not wanna work but mondays are easy days...</td>
      <td>1</td>
      <td>13</td>
      <td>49</td>
      <td>3.769231</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3201</th>
      <td>heromancer   i will take shin</td>
      <td>1</td>
      <td>6</td>
      <td>50</td>
      <td>8.333333</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Remove multiple spaces `"hi   hello    "`


```python
x =  'hi    hello     how are you'
```


```python
' '.join(x.split())
```




    'hi hello how are you'




```python
df['twitts'] = df['twitts'].apply(lambda x: ' '.join(x.split()))
```


```python

```

## Remove HTML tags


```python
!pip install beautifulsoup4
```

    Requirement already satisfied: beautifulsoup4 in c:\users\chitr\appdata\local\programs\python\python36\lib\site-packages (4.9.3)
    Requirement already satisfied: soupsieve>1.2 in c:\users\chitr\appdata\local\programs\python\python36\lib\site-packages (from beautifulsoup4) (2.2.1)
    

    WARNING: You are using pip version 21.0.1; however, version 21.1.1 is available.
    You should consider upgrading via the 'c:\users\chitr\appdata\local\programs\python\python36\python.exe -m pip install --upgrade pip' command.
    


```python
from bs4 import BeautifulSoup
```


```python
x = '<html><h1> thanks for watching it </h1></html>'
```


```python
x.replace('<html><h1>', '').replace('</h1></html>', '') #not rec
```




    ' thanks for watching it '




```python
BeautifulSoup(x, 'lxml').get_text().strip()
```


    ---------------------------------------------------------------------------

    FeatureNotFound                           Traceback (most recent call last)

    <ipython-input-187-2e9db3c14738> in <module>
    ----> 1 BeautifulSoup(x, 'lxml').get_text().strip()
    

    c:\users\chitr\appdata\local\programs\python\python36\lib\site-packages\bs4\__init__.py in __init__(self, markup, features, builder, parse_only, from_encoding, exclude_encodings, element_classes, **kwargs)
        244                     "Couldn't find a tree builder with the features you "
        245                     "requested: %s. Do you need to install a parser library?"
    --> 246                     % ",".join(features))
        247 
        248         # At this point either we have a TreeBuilder instance in
    

    FeatureNotFound: Couldn't find a tree builder with the features you requested: lxml. Do you need to install a parser library?



```python
%%time
df['twitts'] = df['twitts'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text().strip())
```


```python

```

## Remove Accented Chars 


```python
x = 'Áccěntěd těxt'
```


```python
import unicodedata
```


```python
def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x
```


```python
remove_accented_chars(x)
```


```python
df['twitts'] = df['twitts'].apply(lambda x: remove_accented_chars(x))
```


```python

```


```python

```


```python

```

## Remove Stop Words 


```python
x = 'this is a stop words'
```


```python
' '.join([t for t in x.split() if t not in stopwords])
```


```python
df['twitts_no_stop'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))
```


```python
df.sample(5)
```


```python

```


```python

```


```python

```


```python

```

## Convert into base or root form of word 


```python
nlp = spacy.load('en_core_web_sm')
```


```python
x = 'this is chocolates. what is times? this balls'
```


```python
def make_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)
```


```python
make_to_base(x)
```


```python
df['twitts'] = df['twitts'].apply(lambda x: make_to_base(x))
```


```python
df.sample(5)
```


```python

```

## Common words removal 


```python
x = 'this is this okay bye'
```


```python
text = ' '.join(df['twitts'])
```


```python
len(text)
```


```python
text = text.split()
```


```python
len(text)
```


```python
freq_comm = pd.Series(text).value_counts()
```


```python
f20 = freq_comm[:20]
```


```python
f20
```


```python
df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in f20]))
```


```python
df.sample(5)
```


```python

```

## Rare words removal 


```python
rare20 = freq_comm.tail(20)
```


```python
df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare20]))
```


```python
df.sample(5)
```

## Word Cloud Visualization 


```python
# !pip install wordcloud
```


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
text = ' '.join(df['twitts'])
```


```python
len(text)
```


```python
wc = WordCloud(width=800, height=400).generate(text)
plt.imshow(wc)
plt.axis('off')
plt.show()
```


```python

```

## Spelling Correction 


```python
!pip install -U textblob
```


```python
!python -m textblob.download_corpora
```


```python
from textblob import TextBlob
```


```python
x = 'thankks forr waching it'
```


```python
x = TextBlob(x).correct()
```


```python
x
```

## Tokenization using TextBlob



```python
x = 'thanks#watching this video. please like it'
```


```python
TextBlob(x).words
```


```python
doc = nlp(x)
for token in doc:
    print(token)
```


```python

```

## Detecting Nouns 


```python
x = 'Breaking News: Donal Trump, the president of the USA is looking to sign a deal to mine the moon'
```


```python
doc = nlp(x)
```


```python
for noun in doc.noun_chunks:
    print(noun)
```


```python

```

## Language Translation and Detection

Language Code: https://www.loc.gov/standards/iso639-2/php/code_list.php


```python
x
```


```python
tb = TextBlob(x)
```


```python
tb.detect_language()
```


```python
tb.translate(to = 'zh')
```


```python

```

## Use TextBlob's Inbuilt Sentiment Classifier 


```python
from textblob.sentiments import NaiveBayesAnalyzer
```


```python
x = 'we all stands together. we are gonna win this fight'
```


```python
tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
```


```python
tb.sentiment
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
