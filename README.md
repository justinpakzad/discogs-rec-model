# Discogs Rec Model
This repository houses the code related for training the discogs-rec model. It utilizes Spotify's [Annoy](https://github.com/spotify/annoy) library and is trained on data from the Discogs data dumps, as well as additional data such as user wants, haves, and pricing collected via webscraping. At the moment Discogs Rec only supports recommendations for electronic music and hip-hop listed on Discogs. If you would like to use the model via API, please see this repo here: (insert link). If you can't be bothered to setup the API I provided a demo script in the `/demo` folder which will allow you to get predictions from a discogs URL right  away (see example below).

## Repository Structure
- `discogs_rec/`: Contains the source code for the creation of the Recommender System. This includes the preprocessing of the features used to build the model, generating the .ann file, and creating the mappings between release ids and artist/track titles.
- `data/`: Directory for the training data, annoy index, config, and mappings.
- `demo/`: Directory where the demo script lives to fetch recommendations.
- `tests`: Unit tests
## Setup
1. Clone the repository:
```bash
git clone https://github.com/justinpakzad/discogs-rec-model
```
2. Install requirements (only necessary if you want to try the demo, otherwise Docker handles it)
```bash
pip install requirements.txt
```

## Docker Commands 
To build and run the `discogs_rec_model` services with `docker compose`, use the following commands:
```bash
docker compose up --build discogs_rec_model
```
Or to specify a subset of features to be used:
```bash
docker compose run --build discogs_rec_model python main.py --features low median high countries styles wants haves
```

## Demo Example
```bash
python demo/demo.py --url "https://www.discogs.com/release/335130-FL-Untitled"
```
**Returns:**
```json
[
  {
    "artist": "Leftfield",
    "title": "More Than I Know And Not Forgotten (Hard Hands Mix)",
    "url": "https://www.discogs.com/release/1685526"
  },
  {
    "artist": "Joy (15)",
    "title": "Ain't No Sunshine",
    "url": "https://www.discogs.com/release/21667432"
  },
  {
    "artist": "Leftfield",
    "title": "More Than I Know (Remixes)",
    "url": "https://www.discogs.com/release/705478"
  },
  {
    "artist": "Various",
    "title": "Freezone 5",
    "url": "https://www.discogs.com/release/17489977"
  },
  {
    "artist": "Maree",
    "title": "Come To Me",
    "url": "https://www.discogs.com/release/8481752"
  }
]
```
## Notes
Feel free to experiment with the number of features used to generate the Annoy index file. I've experimented with using slightly fewer features and found it resulted in more random but interesting recommendations. You can also adjust the weight of the feature matrices for varied results in weights_dict which resides in the `preprocessing.py` file. Lastly, tweaking the number of trees (`n_trees`) in the Annoy file can enhance recommendation accuracy at the cost of a larger `.ann` file and increased memory usage during queries, as found in `utils.py`. If you are having issues running this locally due to a compute issue, consider lowering the number of trees. This is an ongoing project, I will be trying to add/update new features and fix any bugs that may appear along the way. 






