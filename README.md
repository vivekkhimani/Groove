## Groove: An Unsupervised, Data Sensitive Music Recommendation Algorithm

### VIEW:
- <a href="https://devpost.com/software/groove-wmhrj4?ref_content=user-portfolio&ref_feature=in_progress">Devpost</a>

### IDEA
An unsupervised approach to music recommendation without accessing other users data and use traditional recommendation algorithms like collaborative filtering.

### DESCRIPTION
Spotify and other popular music apps rely heavily on other users data to tailor the best recommendations from millions of options available. Traditionally, they find the music in your playlist and find other users with similar interests and try to recommend music from similar playlists which might not have been a part of your playlist. There might be other instrinsic details in the algorithm, but we are quite sure that this is how they filter the top choices. However, it
might not be possible to use such an approach without having access to other users data. Also, using the private data raises privacy concerns.

### MOTIVATION/SOLUTION
We believe that the existing recommender algorithms definitely do a great job but there are a few loopholes that could be improved:
- <b>Access to other users data:</b> It's dangerous to assume that we will always have access to other users data. With the growing concerns around privacy and security, using the private user data isn't always the best thing to do. Especially, music doesn't seem like a private data, but it can reveal a lot about users mood. For example, a depressed person might resort to sad songs.
- <b>Personalization:</b> If two or more users have same taste for a specific song, it's not neccessary that they might have same taste for other songs. At times, the user might only like a specific song due to artist, albums, or even the actors. In that case, assuming the similar taste of different users introduces a bias which might ruin the recommender performance. Also, we believe that users usually tend to like songs with intrinsic sound characteristics like beats, pitch, timbre,
    tempo, etc.

As a result, we plan to build a recommender system which doesn't rely on other users data and makes recommendations based on the intrinsic sound characteristics like beats, pitch, timbre, tempo, etc. We don't neccessarily claim to beat the existing recommendation performance by Spotify or any other app as they would have put a decent amount of work to come up with their current algorithm. However, we definitely propose that this new recommendation approach would be able to make fairly
diverse recommendations through intrinsic sound characteristics and we will also not use other users data.

### HOW DOES SPOTIFY WORK 
- <b>Source:</b>: https://qz.com/571007/the-magic-that-makes-spotifys-discover-weekly-playlists-so-damn-good/#:~:text=Spotify%20begins%20by%20looking%20at,the%20common%20currency%20on%20Spotify.)
- They primarily use collaborative filtering to create user groups and come up with top candidate songs.
- They will never exactly disclose the detail but we can sell the product based on the limited availability of the data.

### DELIVERABLE
- <b>Web Interface:</b> It should allow user to sign in using Spotify credentials and we will fetch the top songs using Personalization API. Based on the top songs, we will make unsupervised recommendations. Users will be allowed to approve or disapprove of our recommendation and based on their rating, we will improve. More Details under Deployment Plan Section.
- <b>Flask Server:</b> Python-based server which will allow us to use ML time series tools and libraries. We will integrate it with Spotify API and provide our custom endpoints which will be called from the frontend for fetching content.

### OUR PLAN:
- So our dataset consists of English songs, from top X artists, and X, Y, Z genre. We generated the dataset using the Spotify API - Audio Analysis feature.
- Using our dataset, we filtered the key features that might be useful for our unsupervised ML algorithm using custom feature engineering approaches.
- Through our web interface, we take user inputs about top 5 songs from Spotify. The only constraint is that the songs should be available on Spotify so that we can generate the feature map for their preferences.
- Based on the feature map, we run our ML algorithm and pick top 20 scores based on their choices.
- We generate a "clustering plot" to show how our recommendations align with our choices and display it on the website.
- We can also display confidence scores along with our recommendations.
- We can ask users to approve or disapprove of our choices and based on their feedback, we can further tailor the algorithm.
- As we don't have access to the users data, we can add a disclaimer - "We don't claim that you might have heard these songs before as we don't have access to that data, but we definitely recommend giving it a shot."

### ML:
- Unsupervised approach - TimeSeriesKMeans (lib_tsLearn)
- Need to do extensive feature engineering

### DEPLOYMENT:
- Backend: REST API on Flask Server so we can leverage Spotify API + TSLearn (Local Port 3000 exposed on a public IP using ngrok)
- Frontend: HTML, Bootrstap, CSS

- User logs in through the Spotify Authorization (we need to check permissions for this)
- Use Spotify Personalization API to get top 10-20 songs
- Based on the retrieved songs, recommend songs from our trained clusters with confidence score
- Store the results on blockchain for immutability and transparency (we can or cannot implement this and just finesse. depends on blockchain API compatibility with Flask/Python)
- Display the songs in sorted order, cluster graph, and confidence scores
- Provide an interface for user to approve or disapprove and submit
- That's it! :)

### RECOMMENDATION BASIS:
We can provide options for recommending based on the following parameters:
- Top songs and artists (Spotify - Personalization API)
- Based on any of the playlists (Spotify - Playlist API)
- Recently Played Tracks (Spotify - Player API)
- Saved Tracks (Spotify - Library API)

### DATA:
As we didn't have enough resources to compile every song available in the Spotify library as it would add up to 50 million songs, we have used a dataset with 170000 songs (from 1928 - 2020). The dataset can be found at: https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks. As the dataset doesn't include every song in the library, it's possible that our system doesn't include a few songs in the recommendation. However, when we will take this to the production level, we
expect to use a much larger database.
