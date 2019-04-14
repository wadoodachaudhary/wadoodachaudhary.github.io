const BASE_MODEL_URLS = {
    model: 'model_base/model.json',
    metadata: 'model_base/metadata.json',
    image: 'model_base/model_acc.png'
};

const RNN_MODEL_URLS = {
  model: 'model_rnn/model.json',
  metadata: 'model_rnn/metadata.json',
  image: 'model_rnn/model_acc.png'
};

MODEL_URLS = BASE_MODEL_URLS
const examples = {
  'Book1':'He was quite a new man in the circle of the nobility of the Russia.',
  'Book2':'The ladies of Longbourn soon waited on those of Netherfield.',
  'Book3':'Not long after visiting his mother grave Alyosha suddenly announced that he wanted to enter the monastery, and that the monks were willing to receive him as a novice.',
  'Book4':'The deeper that sorrow carves into your being, the more joy you can contain.'
};

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON,img) {
  const img_acc = document.getElementById('ModelAccuracy')
  img_acc.src =img
  img_acc.style.width = "600px";
  img_acc.style.height = "300px";
  document.getElementById('vocabularySize').textContent =metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =metadataJSON['max_len'];

}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  score_string = "Class scores: ";
  for (var x in result.score) {
    score_string += x + " ->  " + result.score[x].toFixed(3) + ", "
  }
  //console.log(score_string);
  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  //settextField(examples['Book1'], predict);
}

function execPrepUI(predict) {
      setPredictFunction(predict);
      doPredict(predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    status('Done loading pretrained model.');
    //disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    this.metadata = await loadHostedMetadata(this.urls.metadata);
    this.image = this.urls.image;
    //showMetadata(metadata);
    this.maxLen = this.metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = this.metadata['word_index']
    //await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    showMetadata(this.metadata,this.image);
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, i);
      //console.log(word, this.wordIndex[word], inputBuffer);
    }
    const input = inputBuffer.toTensor();
    //console.log(input);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};


async function setup() {

  if (await urlExists(MODEL_URLS.model)) {
      predictor = await new Classifier().init(MODEL_URLS);

    status('Model available: ' + MODEL_URLS.model);
    const button_load_model = document.getElementById('load-model');
    button_load_model.addEventListener('click', async () => {
      //document.getElementById('ModelAccuracy').src ='model_js/model_acc.jpg';
      //  alert(MODEL_URLS)
      predictor = await new Classifier().init(MODEL_URLS);
      predictor.loadMetadata()
      //prepUI(x => predictor.predict(x));
    });
    const button_predict = document.getElementById('predict');
    button_predict.addEventListener('click', async () => {
      //const predictor = await new Classifier().init(MODEL_URLS);
      //predictor.loadMetadata()
      execPrepUI(x => predictor.predict(x));
      
    });
    const textField = document.getElementById('text-entry');
    textField.value = examples['Book1'];
    const modelSelect = document.getElementById('model-select');
    modelSelect.addEventListener('change', () => {
         if (modelSelect.value=='model_base')
             MODEL_URLS = BASE_MODEL_URLS
         else
             MODEL_URLS = RNN_MODEL_URLS

    });
    const testExampleSelect = document.getElementById('example-select');
    testExampleSelect.addEventListener('change', () => {
        const textField = document.getElementById('text-entry');
        textField.value = examples[testExampleSelect.value];

    })

    button_predict.style.display = 'inline-block';
    button_load_model.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();