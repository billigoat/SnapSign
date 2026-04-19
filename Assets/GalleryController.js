// @input Asset.VoiceMLModule vmlModule
// @input Component.Text text
// @input SceneObject[] images
// @input Asset.Texture[] aslTextures
// @input float secondsPerLetter = 0.75 {"label": "Seconds Per Letter"}

var wordQueue = [];
var isDisplaying = false;

var options = VoiceML.ListeningOptions.create();
options.shouldReturnAsrTranscription = true;
options.shouldReturnInterimAsrTranscription = true;
options.languageCode = 'en_US';

// --- 1. THE ASL CORE LOGIC ---
function showWord(word) {
    word = word.toUpperCase().replace(/[^A-Z]/g, "");

    for (var i = 0; i < script.images.length; i++) {
        if (i >= word.length) {
            script.images[i].enabled = false;
            continue;
        }

        var char = word.charAt(i);
        var index = char.charCodeAt(0) - "A".charCodeAt(0);

        if (index < 0 || index >= script.aslTextures.length) {
            script.images[i].enabled = false;
            continue;
        }

        script.images[i].enabled = true;
        var image = script.images[i].getComponent("Component.Image");

        if (image && script.aslTextures[index]) {
            var uniqueMat = image.mainMaterial.clone();
            uniqueMat.mainPass.baseTex = script.aslTextures[index];
            image.mainMaterial = uniqueMat;
        }
    }
    
    return word.length; // Return length for timing calculation
}

// --- 2. THE DYNAMIC QUEUE MANAGER ---
function processQueue() {
    if (wordQueue.length === 0) {
        isDisplaying = false;
        // Optional: clear images after the last word is done
        // showWord(""); 
        return;
    }

    isDisplaying = true;
    var nextWord = wordQueue.shift(); 
    
    // Display the word and get its length
    var wordLength = showWord(nextWord);

    // Calculate dynamic delay: Length * 0.75
    // Example: "CAT" = 2.25s | "HELLO" = 3.75s
    var dynamicDelay = wordLength * script.secondsPerLetter;

    var delayEvent = script.createEvent("DelayedCallbackEvent");
    delayEvent.bind(processQueue);
    delayEvent.reset(dynamicDelay);
}

// --- 3. VOICE ML LOGIC ---
var onUpdate = (eventArgs) => {
    var transcription = eventArgs.transcription.trim();
    if (transcription == "") return;

    script.text.text = transcription;

    if (eventArgs.isFinalTranscription) {
        var allWords = transcription.split(" ");
        
        // Reset queue for the most recent 3 words
        wordQueue = [];

        var startIndex = Math.max(0, allWords.length - 3);
        for (var i = startIndex; i < allWords.length; i++) {
            if (allWords[i].length > 0) {
                wordQueue.push(allWords[i]);
            }
        }

        if (!isDisplaying) {
            processQueue();
        }
    }
};

// --- 4. SETUP EVENTS ---
script.vmlModule.onListeningUpdate.add(onUpdate);
script.vmlModule.onListeningEnabled.add(() => script.vmlModule.startListening(options));
script.vmlModule.onListeningDisabled.add(() => script.vmlModule.stopListening());

script.createEvent("OnStartEvent").bind(() => showWord(""));