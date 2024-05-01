# Music-Genre-Classification-Recommendation.io


## Team Members
Shaun Noronha, Sanjeeth Chakkrasali, Khushita Joshi, Alessandra Serpes, Sankalp Vyas

## Background and Motivation
Our project, the Advanced Music Genre Classification System, is inspired by the successes of platforms like Spotify, which utilize machine learning not only to enhance music discovery but also to personalize user experiences. By developing a system that incorporates cutting-edge audio analysis and machine learning models, we aim to improve the accuracy and efficiency of music genre classification. Additionally, by integrating mood recognition technology, our system will offer personalized music recommendations, enriching the user's listening experience based on their current emotional state.

## Dataset
### GTZAN Dataset - Music Genre Classification
We used the data available on [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

The GTZAN dataset is a collection of 1000 audio tracks equally distributed across 10 genres, making it one of the most commonly used datasets for evaluating music genre classification systems. Each track is 30 seconds long, providing a substantial variety of samples for training deep learning models.

Our Datasets contain 10 genres:-

Blues
Classical
Country
Disco
Hip-hop
Jazz
Metal
Pop
Reggae
Rock

## Overview
The GTZAN Music Genre Classification dataset, available on Kaggle, contains audio features extracted from segmented 3-second clips of music tracks. These features provide insights into various aspects of the audio signals and are commonly used in music genre classification tasks.

## Features
Here is a brief description of each feature included in the dataset:

1. **Chroma STFT (Short-Time Fourier Transform) Mean:** Measures the intensity of different pitches in a music track, providing a representation of the audio in terms of its harmonic content across 12 different pitch classes.
2. **RMS (Root Mean Square) Mean:** Indicates the average power or loudness of the audio signal.
3. **Spectral Centroid Mean:** Represents the "center of mass" of the spectrum, giving a sense of the brightness of a sound.
4. **Spectral Bandwidth Mean:** Measures the width of the band of light at half the peak maximum and effectively indicates the range of frequencies present in the sound.
5. **Rolloff Mean:** The frequency below which a specified percentage (typically 85% to 95%) of the total spectral energy lies, highlighting the shape of the audio spectrum.
6. **Zero Crossing Rate Mean:** The rate at which the signal changes signs, which can indicate the noisiness or the complexity of a sound.
7. **Harmony Mean:** Extracts the harmonic components of the audio, which are important for the perception of musical notes.
8. **Perceptual Sharpness Mean:** Measures the sharpness or brightness of the audio, which affects how listeners perceive the "edge" or clarity of a sound.
9. **Tempo:** The speed at which a piece of music is played, calculated in beats per minute (BPM).
10. **MFCC (Mel Frequency Cepstral Coefficients) Mean:** Describes the overall shape of the spectral envelope and is widely used in audio signal processing and speech recognition for timbre and speech clarity characterization.


## Exploratory Data Analysis
1. **BPM Boxplot for Genres**
![Boxplot](Boxplot.png)
The boxplot highlights varied BPM distributions among genres, suggesting tempo as a key feature in genre classification.
Blues and Jazz: Both show a wide range of BPMs with several outliers, indicating diverse substyles within each genre that range from slow to fast tempos.
Classical and Reggae: These genres exhibit narrower BPM ranges, which align with their more consistent and genre-specific tempos.
Country and Pop: Both have moderate BPM ranges around 120, typical for genres that balance slow ballads with faster dance tracks.
Disco and Metal: These genres stand out with higher medians and wider ranges, reflecting their energetic and fast-paced natures.
Hip Hop and Rock: Display moderate to wide BPM distributions, accommodating a variety of subgenres from laid-back to aggressive styles.

3. **Correlation Heatmap (for the MEAN variables)**
![Correlation](Correlation.png)
This correlation heatmap showcases the relationships between various mean variables extracted for music genre classification. Here are key observations:
Strong Interrelationships: Many of the Mel-Frequency Cepstral Coefficients (MFCCs), which are critical for capturing the timbre of audio signals, exhibit moderate to strong correlations with each other. This suggests redundancy in the information they provide, which may impact model complexity and effectiveness.
Lower Correlations with Non-MFCC Features: Features such as chroma_stft_mean, rms_mean, and spectral_centroid_mean show lower correlations with MFCCs. These differences indicate that combining these features with MFCCs could provide complementary information, enhancing the genre classification model's accuracy.
RMS and Perceptual Features: The root mean square (rms_mean), which measures the audio signal's power, and the perceptual mean (percept_mean), likely relating to perceived loudness or clarity, also show limited correlation with MFCCs. Including these could be vital for understanding dynamics and clarity, which are genre-defining characteristics.
Spectral Features' Interaction: The spectral centroid and bandwidth (spectral_centroid_mean, spectral_bandwidth_mean), which reflect the spectral "center of mass" and the width of the spectral energy distribution, respectively, demonstrate a noteworthy relationship. This relationship is important as it could help in distinguishing genres based on their spectral content.

5. **PCA on Genres**
![PCA](PCA.jpg)
The Principal Component Analysis (PCA) plot displayed here illustrates the distribution of music genres based on the reduction of multiple audio feature dimensions into two principal components. Key observations from this PCA on genres are as follows:

Distinct Clusters: Several genres, particularly classical and metal, form distinct clusters, indicating that their audio features are notably different from other genres. This suggests that PCA can effectively reduce feature space while retaining significant genre-specific characteristics.
Overlap Among Genres: Genres like rock, pop, and disco show considerable overlap. This could be due to similarities in their musical structure, such as beat, tempo, and instrumentation, which are not fully separable by the first two principal components.
Spread and Density: The spread and density of points within each genre vary. Jazz and blues, for example, exhibit a broad dispersion, suggesting a higher intra-genre variability compared to more tightly clustered genres like classical.
Potential for Classification: The clear separation of some genre clusters from others provides a promising basis for classification models. However, the overlapping areas will likely require more sophisticated, possibly non-linear, models to achieve high accuracy.


## Implementation
We used the data for our model as follows:

- **Training Data:** 80% of the Data.
- **Test Data:** 20% of the Data.

For the demo, we only considered less than 1% for quick inference.

### Bidirectional LSTM
Bidirectional LSTM (BiLSTM) is a recurrent neural network used primarily for natural language processing. Unlike standard LSTM, the input flows in both directions, and it’s capable of utilizing information from both sides. It’s also a powerful tool for modeling the sequential dependencies between words and phrases in both directions of the sequence.

![BiLSTM](bilstm-1.jpg)

### Nearest-Neighbors
A nearest-neighbors model is a technique used for searching and retrieving similar items or data points from a dataset based on their similarity to a query item. It operates on the principle that items that are close or similar in a feature space should also be similar in their inherent characteristics or properties.

The nearest-neighbors model works by organizing the dataset into a structure that efficiently retrieves the nearest or most similar items to a given query.

### Annoy
Annoy is a library used for approximate nearest-neighbor search. It's particularly useful when dealing with high-dimensional data and finding nearest neighbors efficiently. The library provides a data structure and algorithms that enable fast approximate searches for nearest neighbors, especially in very large datasets.

### HnswLib
HNSW (Hierarchical Navigable Small World) is a library used for approximate nearest neighbor search, similar to Annoy. It's designed to efficiently locate approximate nearest neighbors in high-dimensional spaces, especially in scenarios where traditional methods might struggle due to the curse of dimensionality.

#### Model Design
The model used is seen below:

![LSTM Model](LSTM.jpg)

## Conclusion & Future Work
1. The performance of the nearest neighbor search may vary across different methods (NearestNeighbors, AnnoyIndex, hnswlib). Comparing their accuracy, efficiency, and recall rates can help identify the most suitable method.
2. Using a sentence tokenizer from the BERT family offers advantages in tokenizing text into meaningful segments. Instead of the current embedding model that learns embeddings from scratch, leveraging BERT-based tokenization can benefit the process.
3. Employing exact nearest neighbor search metrics such as cosine similarity or Euclidean distance could refine the search process.

### GitHub Repository
Here is the link for the [repository](https://github.com/Shaun-Noronha/Music-Genre-Classification-Recommendation.io).

### References
We took inspiration from the following papers and worked on our project:

1. Wang Hongdan., Siti SalmiJamali., Chen Zhengping., Shan Qiaojuan., Ren Le. (2022). An intelligent music genre analysis using feature extraction and
classification using deep learning techniques. Elsevier (2022), [Link](https://doi.org/10.1016/j.compeleceng.2022.107978)
2. Teng Li. (2024). Optimizing the configuration of deep learning models for music genre classification. [Link](https://doi.org/10.1016/j.heliyon.2024.e24892)
```

Feel free to adjust any part according to your preferences or add any additional information you find necessary!
