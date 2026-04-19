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
    // Clean word: Uppercase and remove non-letters
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
    
    return word.length;
}

// --- 2. THE QUEUE MANAGER ---
function processQueue() {
    if (wordQueue.length === 0) {
        isDisplaying = false;
        showWord(""); // Clear hands when done
        return;
    }

    isDisplaying = true;
    var nextWord = wordQueue.shift();
    
    // Display the word and get its length for timing
    var wordLength = showWord(nextWord);

    // Dynamic delay: If word is 4 letters, delay is 3 seconds
    var dynamicDelay = Math.max(0.5, wordLength * script.secondsPerLetter);

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
        
        // Add every word from the sentence to the queue
        for (var i = 0; i < allWords.length; i++) {
            if (allWords[i].length > 0) {
                wordQueue.push(allWords[i]);
            }
        }

        if (!isDisplaying) {
            processQueue();
        }
    }
};

// --- 4. SETUP & ERROR CHECK ---
if (script.vmlModule) {
    script.vmlModule.onListeningUpdate.add(onUpdate);
    script.vmlModule.onListeningEnabled.add(function() {
        script.vmlModule.startListening(options);
    });
    script.vmlModule.onListeningDisabled.add(function() {
        script.vmlModule.stopListening();
    });
} else {
    print("CRITICAL: Please drag a Voice ML Module into the Script Inspector!");
}

script.createEvent("OnStartEvent").bind(function() {
    showWord("");
});
