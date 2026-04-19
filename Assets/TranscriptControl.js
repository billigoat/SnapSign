//@input Asset.VoiceMLModule vmlModule
//@input Component.Text text

//To see transcription, create a text object in the Scene. Connect it to this script. In Asset Browser, add VoiceML module, and also connect it to this script. 

var options = VoiceML.ListeningOptions.create();

options.shouldReturnAsrTranscription = true;        // final text
options.shouldReturnInterimAsrTranscription = true; // 🔥 LIVE text
options.languageCode = 'en_US';

var onListeningEnabled = () => {
    script.vmlModule.startListening(options);
};

var onListeningDisabled = () => {
    script.vmlModule.stopListening();
};

var onUpdate = (eventArgs) => {
    if (eventArgs.transcription.trim() == "") return;

    // 🔥 LIVE updating text
    script.text.text = eventArgs.transcription;

    // Optional: detect final result
    if (eventArgs.isFinalTranscription) {
        print("Final: " + eventArgs.transcription);
    }
};

script.vmlModule.onListeningUpdate.add(onUpdate);
script.vmlModule.onListeningEnabled.add(onListeningEnabled);
script.vmlModule.onListeningDisabled.add(onListeningDisabled);