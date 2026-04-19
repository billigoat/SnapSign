// GestureRecognizer — hand poses + static ASL fingerspelling (heuristic).
// Event: Lens Initialized
//
// Uses joint distances plus palm-frame checks: thumb vs palm normal (A vs thumbs up),
// index-chain spread in the palm plane, and tip separations for H / U / V.
//
// Snapshot: script.lastHandSnapshot / script.lastHandSnapshotJson — world XYZ per joint,
// palm-local coords (u,v,n)/scale, palm euler deg, per-finger curl + tip placement. Use
// snapshotMode + collectHandSnapshotNow() for template matching / debugging.
//
// Setup: GlobalTrackingHelper + HandTrackingController(s), or assign handController.
// Static letters A–Z (heuristic): all have scores; explicit rules for clearer poses.
// J and Z normally use motion — J uses same static pose as I; Z is weak (often confused with D/G).
// Pair with asl_camera_chart.js: assign Device Camera + alphabet texture, highlight grid cell.
//
// Learned classification: set classifierMode to use k-NN (record samples: snapshotMode "Tap record kNN")
// or an MLComponent with ONNX matching input size ASL_POSE_FEATURE_DIM (75) and 28 outputs in class order
// documented in getMlClassOrder() below.

//@ui {"widget":"separator"}
//@input Component.ScriptComponent handController {"label":"HandTrackingController (optional)"}
//@input int watchHand = 0 {"widget":"combobox","values":[{"value":"0","label":"Active hand"},{"value":"1","label":"Left only"},{"value":"2","label":"Right only"},{"value":"3","label":"Both hands"}]}
//@input int minStableFrames = 10 {"label":"Stable frames before save"}
//@input float extendRatio = 1.18 {"label":"Finger extended (strict): wrist→tip / wrist→base"}
//@input float thumbExtendRatio = 1.12 {"label":"Thumb extended: wrist→thumb-3 / wrist→thumb-1"}
//@input float aslBExtendRatio = 1.08 {"label":"ASL B: non-thumb extended ≥ this (can be lower than strict)"}
//@input float aslCMaxStraightRatio = 1.55 {"label":"ASL C: reject if all 4 fingers straighter than this (avoid B→C)"}
//@ui {"widget":"separator"}
//@input float peaceFingerExtendRatio = 1.05 {"label":"asl_v: index/mid extended ≥ this"}
//@input float curlMaxRatio = 1.13 {"label":"Finger curled: wrist→tip / wrist→base ≤ this"}
//@input float peaceCompanionMaxRatio = 1.26 {"label":"asl_v: ring/pinky down if ratio <= this"}
//@input float peaceSpreadRatio = 1.0 {"label":"asl_v: tip spread vs knuckle spread ≥ this"}
//@input float phonePinkyExtendRatio = 1.14 {"label":"asl_y: pinky ratio ≥ this"}
//@ui {"widget":"separator"}
//@input float aslAThumbIndexMaxRatio = 0.92 {"label":"ASL A: thumb-index norm <= this"}
//@input float aslAThumbIndexSoftMargin = 0.12 {"label":"ASL A: soft margin when !thumbs-up metric"}
//@input float aslACurlMaxRatio = 1.17 {"label":"ASL A: finger curl max"}
//@input bool aslAThumbNearerIndexThanRing = true {"label":"ASL A: thumb-3 nearer index-1 than ring-1"}
//@input float aslAThumbOutOfPalmMax = 0.42 {"label":"ASL A: thumb out-of-palm (n) / scale <= this"}
//@input float aslThumbsUpMinOutOfPalm = 0.55 {"label":"thumbs_up: thumb out-of-palm / scale >= this"}
//@ui {"widget":"separator"}
//@input float aslRMaxTipToKnuckle = 0.32 {"label":"ASL R: index/mid tip spread / knuckle spread < this (crossed)"}
//@input float aslHMaxTipToKnuckle = 0.58 {"label":"ASL H: tip/knuckle spread <= this (touching), above R band"}
//@input float aslUMaxTipToKnuckle = 0.98 {"label":"ASL U: tip/knuckle spread <= this (parallel pair), above H"}
//@input float aslOMaxAvgTipDist = 0.95 {"label":"ASL O: avg tip-to-thumb-3 / scale <= this"}
//@ui {"widget":"separator"}
//@input int snapshotMode = 0 {"widget":"combobox","values":[{"value":"0","label":"Snapshot off"},{"value":"1","label":"Snapshot each frame (first active hand)"},{"value":"2","label":"Snapshot on screen tap (print JSON)"},{"value":"3","label":"Tap: record kNN sample"}]}
//@input bool snapshotIncludePalmLocal = true {"label":"Snapshot: palm-local (u,v,n)/scale per joint"}
//@ui {"widget":"separator"}
//@input int classifierMode = 0 {"widget":"combobox","values":[{"value":"0","label":"Heuristic only"},{"value":"1","label":"k-NN + heuristic fallback"},{"value":"2","label":"k-NN only (else unknown)"},{"value":"3","label":"ML component + heuristic fallback"},{"value":"4","label":"ML, then k-NN, then heuristic"}]}
//@input Component.MLComponent mlComponent {"optional":true,"label":"Snap ML: pose classifier (75 floats in)"}
//@input string mlInputName = "input" {"label":"ML input placeholder name"}
//@input string mlOutputName = "output" {"label":"ML output placeholder name"}
//@input float mlMinConfidence = 0.2 {"label":"ML min probability (or post-softmax max)"}
//@input bool mlOutputAreProbabilities = false {"label":"ML output is already probabilities (skip softmax)"}
//@input int knnK = 5 {"label":"k-NN vote size"}
//@input int knnMinTotalSamples = 12 {"label":"k-NN min samples before use"}
//@input float knnMaxDistance = 5.0 {"label":"k-NN reject if nearest farther than this (L2)"}
//@input int knnMaxPerClass = 40 {"label":"k-NN max stored samples per class"}
//@input int knnMaxTotalSamples = 3000 {"label":"k-NN max total samples"}
//@input string knnRecordLabel = "a" {"label":"Letter for tap-record: a–z, fist, thumbs_up"}
//@ui {"widget":"separator"}
//@input bool debugHOnTap = false {"label":"Debug H on tap (two up + rest curled)"}
//@input bool debugHEachFrame = false {"label":"Debug H each frame (verbose)"}
//@input bool debugIOnTap = false {"label":"Debug I on tap (Y-like + tucked thumb)"}
//@input bool debugIEachFrame = false {"label":"Debug I each frame (verbose)"}
//@input float aslHPairMinPalmN = 0.46 {"label":"ASL H: |pair dir·palm N| >= this (horizontal in lens space)"}
//@input float aslUPairMaxPalmN = 0.40 {"label":"ASL U: |pair dir·palm N| <= this (vertical in lens space)"}
//@input float aslHThumbMaxAlongIndexNorm = 1.18 {"label":"ASL H: thumb tucked if min(thumb-3→index*) ≤ this×scale"}
//@input float aslIThumbMaxAlongVersusY = 0.46 {"label":"ASL I: tucked if along-index ≤ this (when not in Y band)"}
//@input float aslYThumbMinAlongVersusI = 0.48 {"label":"ASL Y: thumb out if along-index ≥ this (with t), if ratio gate fails"}
//@input float aslYThumbMinWristRatio = 2.0 {"label":"ASL Y vs I: wrist thumb ratio ≥ this only if along-index also > ratioAlongMin"}
//@input float aslYThumbRatioGateMinAlong = 0.36 {"label":"ASL Y: ratio-out gate ignored if along-index ≤ this (tucked I)"}
//@input float aslEThumbMaxAbsU = 1.05 {"label":"ASL E: thumb palm-local |u| <= this"}
//@input float aslEThumbMaxAbsV = 0.65 {"label":"ASL E: thumb palm-local |v| <= this"}
//@input float aslEThumbMinN = -0.45 {"label":"ASL E: thumb palm-local n >= this"}
//@input float aslEThumbMaxN = 0.55 {"label":"ASL E: thumb palm-local n <= this"}

var WATCH_ACTIVE = 0;
var WATCH_LEFT = 1;
var WATCH_RIGHT = 2;
var WATCH_BOTH = 3;

var store =
    global.persistentStorageSystem && global.persistentStorageSystem.store
        ? global.persistentStorageSystem.store
        : null;
var STORAGE_KEY = "savedGestureType";
var STORAGE_KEY_LEFT = "savedGestureType_left";
var STORAGE_KEY_RIGHT = "savedGestureType_right";
var KNN_STORAGE_KEY = "asl_pose_knn_v1";
var POSE_FEATURE_VERSION = 1;
/** Fixed layout: palm-local (u,v,n) for SNAPSHOT_JOINTS + 5 curl ratios + 6 norm distances + thumb extend ratio. */
var ASL_POSE_FEATURE_DIM = 75;

var stableState = {
    override: { stableCount: 0, lastRawLabel: null, lastCommittedLabel: null },
    left: { stableCount: 0, lastRawLabel: null, lastCommittedLabel: null },
    right: { stableCount: 0, lastRawLabel: null, lastCommittedLabel: null },
    active: { stableCount: 0, lastRawLabel: null, lastCommittedLabel: null }
};

/** Maps classifier label to ASL chart letter A–Z (empty if not a letter). */
function gestureLabelToLetter(label) {
    if (!label || label.length < 5) {
        return "";
    }
    if (label.indexOf("asl_") !== 0) {
        return "";
    }
    var rest = label.substring(4);
    if (rest.length !== 1) {
        return "";
    }
    return rest.toUpperCase();
}

script.lastInstantLabel = "";

function getLeftScript() {
    if (global.leftHand) {
        return global.leftHand();
    }
    return null;
}

function getRightScript() {
    if (global.rightHand) {
        return global.rightHand();
    }
    return null;
}

function getWatchTargets() {
    if (script.handController) {
        try {
            if (!script.handController.isTracking || !script.handController.isTracking()) {
                return [];
            }
        } catch (eH) {
            return [];
        }
        return [{ bucket: "override", script: script.handController, storageKey: STORAGE_KEY, logPrefix: "" }];
    }

    var left = getLeftScript();
    var right = getRightScript();

    switch (script.watchHand) {
        case WATCH_LEFT:
            if (left && left.isTracking()) {
                return [{ bucket: "left", script: left, storageKey: STORAGE_KEY_LEFT, logPrefix: "left: " }];
            }
            return [];
        case WATCH_RIGHT:
            if (right && right.isTracking()) {
                return [{ bucket: "right", script: right, storageKey: STORAGE_KEY_RIGHT, logPrefix: "right: " }];
            }
            return [];
        case WATCH_BOTH:
            var both = [];
            if (left && left.isTracking()) {
                both.push({ bucket: "left", script: left, storageKey: STORAGE_KEY_LEFT, logPrefix: "left: " });
            }
            if (right && right.isTracking()) {
                both.push({ bucket: "right", script: right, storageKey: STORAGE_KEY_RIGHT, logPrefix: "right: " });
            }
            return both;
        default:
            if (global.getActiveHandController) {
                var active = global.getActiveHandController();
                if (active && active.isTracking()) {
                    var side = global.getHand ? global.getHand() : null;
                    var prefix = side === "L" ? "left: " : side === "R" ? "right: " : "";
                    return [{ bucket: "active", script: active, storageKey: STORAGE_KEY, logPrefix: prefix }];
                }
            }
            return [];
    }
}

var SNAPSHOT_JOINTS = [
    "wrist",
    "thumb-0",
    "thumb-1",
    "thumb-2",
    "thumb-3",
    "index-0",
    "index-1",
    "index-2",
    "index-3",
    "mid-0",
    "mid-1",
    "mid-2",
    "mid-3",
    "ring-0",
    "ring-1",
    "ring-2",
    "ring-3",
    "pinky-0",
    "pinky-1",
    "pinky-2",
    "pinky-3"
];

function round4(x) {
    return Math.round(x * 10000) / 10000;
}

function vec3ToObj(v) {
    if (!v) {
        return null;
    }
    return { x: round4(v.x), y: round4(v.y), z: round4(v.z) };
}

function getJointWorld(handScript, jointName) {
    try {
        if (!handScript || !handScript.getJoint) {
            return null;
        }
        var j = handScript.getJoint(jointName);
        if (!j || j.position === undefined || j.position === null) {
            return null;
        }
        return j.position;
    } catch (eJ) {
        return null;
    }
}

function jointDistSnapshot(handScript, a, b) {
    try {
        if (!handScript || !handScript.getJointsDistance) {
            return null;
        }
        return handScript.getJointsDistance(a, b);
    } catch (eD) {
        return null;
    }
}

function buildPalmBasisFromHand(handScript) {
    try {
        var w = getJointWorld(handScript, "wrist");
        var mid0 = getJointWorld(handScript, "mid-0");
        var idx0 = getJointWorld(handScript, "index-0");
        if (!w || !mid0 || !idx0) {
            return null;
        }
        var scale = jointDistSnapshot(handScript, "wrist", "mid-0");
        if (scale === null || scale < 1e-5) {
            scale = 1;
        }
        var u = mid0.sub(w);
        var lenU = u.length;
        if (lenU < 1e-5) {
            return null;
        }
        u = u.uniformScale(1 / lenU);
        var vRaw = idx0.sub(w);
        var vPerpU = vRaw.sub(u.uniformScale(vRaw.dot(u)));
        var lenV = vPerpU.length;
        if (lenV < 1e-5) {
            return null;
        }
        var v = vPerpU.uniformScale(1 / lenV);
        var n = u.cross(v);
        var lenN = n.length;
        if (lenN < 1e-5) {
            return null;
        }
        n = n.uniformScale(1 / lenN);
        return { w: w, u: u, v: v, n: n, scale: scale };
    } catch (eB) {
        return null;
    }
}

function palmLocalPoint(p, basis) {
    if (!p || !basis) {
        return null;
    }
    var rel = p.sub(basis.w);
    return {
        u: round4(rel.dot(basis.u) / basis.scale),
        v: round4(rel.dot(basis.v) / basis.scale),
        n: round4(rel.dot(basis.n) / basis.scale)
    };
}

function fingerSnapshotFeatures(handScript, prefix, basis) {
    if (!basis) {
        return null;
    }
    var sc = basis.scale;
    var p0 = getJointWorld(handScript, prefix + "-0");
    var p3 = getJointWorld(handScript, prefix + "-3");
    if (!p0 || !p3) {
        return null;
    }
    var dBase = basis.w.distance(p0);
    var dTipW = basis.w.distance(p3);
    var curlRatio = dBase > 1e-5 ? round4(dTipW / dBase) : -1;
    var chain = p3.sub(p0);
    var cl = chain.length;
    var boneAlongN = 0;
    if (cl > 1e-5) {
        boneAlongN = round4(chain.uniformScale(1 / cl).dot(basis.n));
    }
    return {
        curlRatio: curlRatio,
        tipPalmLocal: palmLocalPoint(p3, basis),
        boneAlongPalmNormal: boneAlongN,
        chainLengthNorm: round4(cl / sc)
    };
}

function collectHandSnapshot(handScript) {
    if (!handScript || !handScript.isTracking || !handScript.isTracking()) {
        return null;
    }
    var basis = buildPalmBasisFromHand(handScript);
    var jointsWorld = {};
    var jointsPalmLocal = {};
    var ji;
    for (ji = 0; ji < SNAPSHOT_JOINTS.length; ji++) {
        var jn = SNAPSHOT_JOINTS[ji];
        var pos = getJointWorld(handScript, jn);
        jointsWorld[jn] = vec3ToObj(pos);
        if (script.snapshotIncludePalmLocal && basis && pos) {
            jointsPalmLocal[jn] = palmLocalPoint(pos, basis);
        }
    }
    var palmEulerDeg = null;
    var palmQuatStr = null;
    if (basis) {
        try {
            var qP = quat.lookAt(basis.n, basis.v);
            var eu = qP.toEulerAngles().uniformScale(180 / Math.PI);
            palmEulerDeg = { x: round4(eu.x), y: round4(eu.y), z: round4(eu.z) };
            palmQuatStr = qP.toString();
        } catch (eQ) {
            palmEulerDeg = null;
        }
    }
    var fingers = {
        thumb: fingerSnapshotFeatures(handScript, "thumb", basis),
        index: fingerSnapshotFeatures(handScript, "index", basis),
        mid: fingerSnapshotFeatures(handScript, "mid", basis),
        ring: fingerSnapshotFeatures(handScript, "ring", basis),
        pinky: fingerSnapshotFeatures(handScript, "pinky", basis)
    };
    return {
        version: 1,
        scaleWristMid0: basis ? round4(basis.scale) : null,
        wrist: jointsWorld.wrist,
        palmBasis: basis
            ? {
                  u: vec3ToObj(basis.u),
                  v: vec3ToObj(basis.v),
                  n: vec3ToObj(basis.n)
              }
            : null,
        palmEulerDeg: palmEulerDeg,
        palmQuat: palmQuatStr,
        jointsWorld: jointsWorld,
        jointsPalmLocal: jointsPalmLocal,
        fingers: fingers
    };
}

function snapshotToJson(snap) {
    if (!snap) {
        return "";
    }
    try {
        var plain = {
            version: snap.version,
            scaleWristMid0: snap.scaleWristMid0,
            wrist: snap.wrist,
            palmBasis: snap.palmBasis,
            palmEulerDeg: snap.palmEulerDeg,
            palmQuat: snap.palmQuat,
            jointsWorld: snap.jointsWorld,
            jointsPalmLocal: snap.jointsPalmLocal,
            fingers: snap.fingers
        };
        return JSON.stringify(plain);
    } catch (eJ) {
        return "";
    }
}

function refreshHandSnapshotForScript(handScript) {
    var snap = collectHandSnapshot(handScript);
    script.lastHandSnapshot = snap;
    script.lastHandSnapshotJson = snapshotToJson(snap);
    return snap;
}

script.lastHandSnapshot = null;
script.lastHandSnapshotJson = "";

script.collectHandSnapshotNow = function (hs) {
    var h = hs;
    if (!h) {
        var tt = getWatchTargets();
        if (tt.length > 0) {
            h = tt[0].script;
        }
    }
    if (!h) {
        script.lastHandSnapshot = null;
        script.lastHandSnapshotJson = "";
        return null;
    }
    return refreshHandSnapshotForScript(h);
};

script.getHandSnapshot = function () {
    return script.lastHandSnapshot;
};

script.getHandSnapshotJson = function () {
    return script.lastHandSnapshotJson;
};

script.lastMlConfidence = 0;

function getMlClassOrder() {
    return [
        "asl_a",
        "asl_b",
        "asl_c",
        "asl_d",
        "asl_e",
        "asl_f",
        "asl_g",
        "asl_h",
        "asl_i",
        "asl_j",
        "asl_k",
        "asl_l",
        "asl_m",
        "asl_n",
        "asl_o",
        "asl_p",
        "asl_q",
        "asl_r",
        "asl_s",
        "asl_t",
        "asl_u",
        "asl_v",
        "asl_w",
        "asl_x",
        "asl_y",
        "asl_z",
        "fist",
        "thumbs_up"
    ];
}

function fingerRatioMl(handScript, prefix) {
    var dTip = jointDistSnapshot(handScript, "wrist", prefix + "-3");
    var dBase = jointDistSnapshot(handScript, "wrist", prefix + "-0");
    if (dTip === null || dBase === null || dBase < 1e-5) {
        return 0;
    }
    return dTip / dBase;
}

function buildPoseFeatureVector(handScript) {
    if (!handScript || !handScript.isTracking || !handScript.isTracking()) {
        return null;
    }
    var basis = buildPalmBasisFromHand(handScript);
    if (!basis) {
        return null;
    }
    var vec = [];
    var ji;
    for (ji = 0; ji < SNAPSHOT_JOINTS.length; ji++) {
        var jn = SNAPSHOT_JOINTS[ji];
        var pos = getJointWorld(handScript, jn);
        if (!pos) {
            return null;
        }
        var pl = palmLocalPoint(pos, basis);
        if (!pl) {
            return null;
        }
        vec.push(pl.u, pl.v, pl.n);
    }
    var prefs = ["thumb", "index", "mid", "ring", "pinky"];
    for (var fp = 0; fp < prefs.length; fp++) {
        vec.push(round4(fingerRatioMl(handScript, prefs[fp])));
    }
    var pairs = [
        ["index-3", "mid-3"],
        ["index-3", "thumb-3"],
        ["mid-3", "ring-3"],
        ["ring-3", "pinky-3"],
        ["thumb-3", "mid-1"],
        ["index-1", "mid-1"]
    ];
    for (var pp = 0; pp < pairs.length; pp++) {
        var d = jointDistSnapshot(handScript, pairs[pp][0], pairs[pp][1]);
        vec.push(d !== null && basis.scale > 1e-5 ? round4(d / basis.scale) : 0);
    }
    var dTT = jointDistSnapshot(handScript, "wrist", "thumb-3");
    var dTM = jointDistSnapshot(handScript, "wrist", "thumb-1");
    vec.push(dTT !== null && dTM !== null && dTM > 1e-5 ? round4(dTT / dTM) : 0);
    if (vec.length !== ASL_POSE_FEATURE_DIM) {
        return null;
    }
    if (typeof global !== "undefined") {
        global.__aslPoseFeatureDim = ASL_POSE_FEATURE_DIM;
    }
    return vec;
}

function loadKnnDataset() {
    var empty = { v: POSE_FEATURE_VERSION, samples: [] };
    if (!store) {
        return empty;
    }
    try {
        var s = store.getString(KNN_STORAGE_KEY);
        if (!s || s.length < 2) {
            return empty;
        }
        var o = JSON.parse(s);
        if (!o.samples || o.v !== POSE_FEATURE_VERSION) {
            return empty;
        }
        return o;
    } catch (eK) {
        return empty;
    }
}

function saveKnnDataset(data) {
    if (!store) {
        return;
    }
    try {
        store.putString(KNN_STORAGE_KEY, JSON.stringify(data));
    } catch (eS) {
        print("[Gesture] kNN save failed: " + eS);
    }
}

function appendKnnSample(vec, label) {
    var data = loadKnnDataset();
    var maxC = script.knnMaxPerClass > 2 ? script.knnMaxPerClass : 2;
    var cnt = 0;
    var i;
    for (i = 0; i < data.samples.length; i++) {
        if (data.samples[i].label === label) {
            cnt++;
        }
    }
    while (cnt >= maxC) {
        var removed = false;
        for (i = 0; i < data.samples.length; i++) {
            if (data.samples[i].label === label) {
                data.samples.splice(i, 1);
                cnt--;
                removed = true;
                break;
            }
        }
        if (!removed) {
            break;
        }
    }
    data.samples.push({ label: label, vec: vec });
    var maxT = script.knnMaxTotalSamples > 10 ? script.knnMaxTotalSamples : 500;
    while (data.samples.length > maxT) {
        data.samples.shift();
    }
    saveKnnDataset(data);
}

function l2DistanceVec(a, b) {
    var s = 0;
    var i;
    for (i = 0; i < a.length && i < b.length; i++) {
        var d = a[i] - b[i];
        s += d * d;
    }
    return Math.sqrt(s);
}

function knnPredictLabel(vec) {
    var data = loadKnnDataset();
    var minTot = script.knnMinTotalSamples > 0 ? script.knnMinTotalSamples : 8;
    if (data.samples.length < minTot) {
        return null;
    }
    var dists = [];
    var i;
    for (i = 0; i < data.samples.length; i++) {
        var s = data.samples[i];
        if (!s.vec || s.vec.length !== vec.length) {
            continue;
        }
        dists.push({ d: l2DistanceVec(vec, s.vec), label: s.label });
    }
    if (dists.length < 1) {
        return null;
    }
    dists.sort(function (a, b) {
        return a.d - b.d;
    });
    var maxD = script.knnMaxDistance > 0 ? script.knnMaxDistance : 10;
    if (dists[0].d > maxD) {
        return null;
    }
    var k = script.knnK > 0 ? script.knnK : 3;
    if (k > dists.length) {
        k = dists.length;
    }
    var votes = {};
    var j;
    for (j = 0; j < k; j++) {
        var lb = dists[j].label;
        votes[lb] = (votes[lb] || 0) + 1;
    }
    var best = null;
    var bc = 0;
    for (var key in votes) {
        if (votes[key] > bc) {
            bc = votes[key];
            best = key;
        }
    }
    return best;
}

function softmaxArgmax(probs) {
    var m = -1e30;
    var i;
    for (i = 0; i < probs.length; i++) {
        if (probs[i] > m) {
            m = probs[i];
        }
    }
    var sum = 0;
    var ex = [];
    for (i = 0; i < probs.length; i++) {
        ex[i] = Math.exp(probs[i] - m);
        sum += ex[i];
    }
    if (sum < 1e-10) {
        return { i: 0, p: 0 };
    }
    var bestI = 0;
    var bestP = 0;
    for (i = 0; i < probs.length; i++) {
        var p = ex[i] / sum;
        if (p > bestP) {
            bestP = p;
            bestI = i;
        }
    }
    return { i: bestI, p: bestP };
}

function mlComponentPredictLabel(vec) {
    script.lastMlConfidence = 0;
    if (!script.mlComponent) {
        return null;
    }
    try {
        var inp = script.mlComponent.getInput(script.mlInputName);
        if (!inp || !inp.data) {
            return null;
        }
        var d = inp.data;
        var n = Math.min(vec.length, d.length);
        var i;
        for (i = 0; i < n; i++) {
            d[i] = vec[i];
        }
        for (i = n; i < d.length; i++) {
            d[i] = 0;
        }
        script.mlComponent.runImmediate(true);
        var outPh = script.mlComponent.getOutput(script.mlOutputName);
        if (!outPh || !outPh.data) {
            return null;
        }
        var od = outPh.data;
        var len = od.length;
        if (len < 1) {
            return null;
        }
        var arr = [];
        for (i = 0; i < len; i++) {
            arr.push(od[i]);
        }
        var bestI = 0;
        var bestP = 0;
        if (script.mlOutputAreProbabilities) {
            for (i = 0; i < arr.length; i++) {
                if (arr[i] > bestP) {
                    bestP = arr[i];
                    bestI = i;
                }
            }
        } else {
            var sm = softmaxArgmax(arr);
            bestI = sm.i;
            bestP = sm.p;
        }
        script.lastMlConfidence = bestP;
        var minP = script.mlMinConfidence > 0 ? script.mlMinConfidence : 0.05;
        if (bestP < minP) {
            return null;
        }
        var order = getMlClassOrder();
        if (bestI < 0 || bestI >= order.length) {
            return null;
        }
        return order[bestI];
    } catch (eM) {
        print("[Gesture] ML inference: " + eM);
        return null;
    }
}

function resolveLearnedClassification(handScript, heuristicLabel) {
    // Keep explicit E when heuristic rules are confident; avoid ML/kNN overriding it.
    if (
        heuristicLabel === "asl_e" ||
        heuristicLabel === "asl_g" ||
        heuristicLabel === "asl_h" ||
        heuristicLabel === "asl_i" ||
        heuristicLabel === "asl_y"
    ) {
        return heuristicLabel;
    }
    var mode = script.classifierMode;
    if (mode === 0) {
        return heuristicLabel;
    }
    var vec = buildPoseFeatureVector(handScript);
    if (!vec) {
        return heuristicLabel;
    }
    if (mode === 3 || mode === 4) {
        var mlp = mlComponentPredictLabel(vec);
        if (mlp) {
            return mlp;
        }
    }
    if (mode === 1 || mode === 2 || mode === 4) {
        var knp = knnPredictLabel(vec);
        if (knp) {
            return knp;
        }
        if (mode === 2) {
            return "unknown";
        }
    }
    return heuristicLabel;
}

function recordKnnSampleFromTap() {
    var tt = getWatchTargets();
    if (tt.length < 1) {
        print("[Gesture] kNN record: no tracked hand");
        return;
    }
    var vec = buildPoseFeatureVector(tt[0].script);
    if (!vec) {
        print("[Gesture] kNN record: could not build pose vector");
        return;
    }
    var letter = (script.knnRecordLabel || "a").toLowerCase().trim();
    var label;
    if (letter === "fist") {
        label = "fist";
    } else if (letter === "thumbs_up" || letter === "thumbsup") {
        label = "thumbs_up";
    } else if (letter.length === 1 && letter >= "a" && letter <= "z") {
        label = "asl_" + letter;
    } else {
        print("[Gesture] kNN record: knnRecordLabel must be a-z, fist, or thumbs_up");
        return;
    }
    appendKnnSample(vec, label);
    print("[Gesture] kNN recorded " + label + " (n=" + loadKnnDataset().samples.length + ")");
}

function debugHForHand(handScript, prefix) {
    if (!handScript) {
        print("[Gesture][H debug] no hand script");
        return;
    }
    var pfx = prefix ? prefix : "";
    function jdDbg(a, b) {
        try {
            if (!handScript.getJointsDistance) {
                return null;
            }
            return handScript.getJointsDistance(a, b);
        } catch (e) {
            return null;
        }
    }
    function wpos(jn) {
        return getJointWorld(handScript, jn);
    }
    function fingerRatioDbg(name) {
        var dTip = jdDbg("wrist", name + "-3");
        var dBase = jdDbg("wrist", name + "-0");
        if (dTip === null || dBase === null || dBase < 1e-5) {
            return -1;
        }
        return dTip / dBase;
    }
    var ri = fingerRatioDbg("index");
    var rm = fingerRatioDbg("mid");
    var rr = fingerRatioDbg("ring");
    var rp = fingerRatioDbg("pinky");
    var rt = fingerRatioDbg("thumb");
    var ext = script.extendRatio;
    var curlMax = script.curlMaxRatio;
    var iExt = ri >= ext;
    var mExt = rm >= ext;
    var rCurl = rr >= 0 && rr <= curlMax;
    var pCurl = rp >= 0 && rp <= curlMax;
    var nUp = (iExt ? 1 : 0) + (mExt ? 1 : 0) + (rr >= ext ? 1 : 0) + (rp >= ext ? 1 : 0);
    var scale = jdDbg("wrist", "mid-0");
    if (scale === null || scale < 1e-5) {
        scale = 1;
    }
    var d0 = jdDbg("thumb-3", "index-0");
    var d1 = jdDbg("thumb-3", "index-1");
    var d2 = jdDbg("thumb-3", "index-2");
    var bestAlong = 999;
    if (d0 !== null) {
        bestAlong = Math.min(bestAlong, d0);
    }
    if (d1 !== null) {
        bestAlong = Math.min(bestAlong, d1);
    }
    if (d2 !== null) {
        bestAlong = Math.min(bestAlong, d2);
    }
    var alongN = bestAlong < 900 ? bestAlong / scale : -1;
    var maxAlong = script.aslHThumbMaxAlongIndexNorm > 0 ? script.aslHThumbMaxAlongIndexNorm : 1.18;
    var tCurl = rt >= 0 && rt <= curlMax;
    var thumbTucked = tCurl || (alongN >= 0 && alongN <= maxAlong);
    var basis = buildPalmBasisFromHand(handScript);
    var palmN = -1;
    var worldV = -1;
    var i0 = wpos("index-0");
    var i3 = wpos("index-3");
    var m0 = wpos("mid-0");
    var m3 = wpos("mid-3");
    if (basis && i0 && i3 && m0 && m3) {
        var di = i3.sub(i0);
        var dm = m3.sub(m0);
        var dir = di.add(dm);
        var len = dir.length;
        if (len > 1e-5) {
            dir = dir.uniformScale(1 / len);
            palmN = Math.abs(dir.dot(basis.n));
            var up = new vec3(0, 1, 0);
            worldV = Math.abs(dir.dot(up));
        }
    }
    var minPalmNH = script.aslHPairMinPalmN > 0 ? script.aslHPairMinPalmN : 0.46;
    var palmOkH = palmN >= 0 && palmN >= minPalmNH;
    var looksLikeH = iExt && mExt && nUp === 2 && rCurl && pCurl && thumbTucked && palmOkH;
    print(
        "[Gesture][H debug] " +
            pfx +
            "extendRatio=" +
            ext.toFixed(3) +
            " curlMaxRatio=" +
            curlMax.toFixed(3) +
            " minPalmN(H)=" +
            minPalmNH.toFixed(3) +
            " |pair·N|=" +
            (palmN < 0 ? "?" : palmN.toFixed(3)) +
            " |pair·worldUp|=" +
            (worldV < 0 ? "?" : worldV.toFixed(3)) +
            " thumbAlongN=" +
            (alongN < 0 ? "?" : alongN.toFixed(3)) +
            " maxAlong=" +
            maxAlong.toFixed(3) +
            " ratios i/m/r/p/t=" +
            (ri < 0 ? "?" : ri.toFixed(3)) +
            "/" +
            (rm < 0 ? "?" : rm.toFixed(3)) +
            "/" +
            (rr < 0 ? "?" : rr.toFixed(3)) +
            "/" +
            (rp < 0 ? "?" : rp.toFixed(3)) +
            "/" +
            (rt < 0 ? "?" : rt.toFixed(3)) +
            " i/m ext=" +
            iExt +
            "/" +
            mExt +
            " nFingersStrictEquiv=" +
            nUp +
            " r/p curled=" +
            rCurl +
            "/" +
            pCurl +
            " thumbTucked=" +
            thumbTucked +
            " palmOkH=" +
            palmOkH +
            " => looksLikeAslH=" +
            looksLikeH
    );
}

function debugIForHand(handScript, prefix) {
    if (!handScript) {
        print("[Gesture][I debug] no hand script");
        return;
    }
    var pfx = prefix ? prefix : "";
    function jdDbg(a, b) {
        try {
            if (!handScript.getJointsDistance) {
                return null;
            }
            return handScript.getJointsDistance(a, b);
        } catch (e) {
            return null;
        }
    }
    function wpos(jn) {
        return getJointWorld(handScript, jn);
    }
    function fingerRatioDbg(name) {
        var dTip = jdDbg("wrist", name + "-3");
        var dBase = jdDbg("wrist", name + "-0");
        if (dTip === null || dBase === null || dBase < 1e-5) {
            return -1;
        }
        return dTip / dBase;
    }
    function bendAmount(a, b, c) {
        var pa = wpos(a);
        var pb = wpos(b);
        var pc = wpos(c);
        if (!pa || !pb || !pc) {
            return -1;
        }
        var v1 = pa.sub(pb);
        var v2 = pc.sub(pb);
        var l1 = v1.length;
        var l2 = v2.length;
        if (l1 < 1e-5 || l2 < 1e-5) {
            return -1;
        }
        v1 = v1.uniformScale(1 / l1);
        v2 = v2.uniformScale(1 / l2);
        var d = v1.dot(v2);
        if (d > 1) {
            d = 1;
        }
        if (d < -1) {
            d = -1;
        }
        var ang = Math.acos(d);
        return (Math.PI - ang) / Math.PI;
    }
    var ri = fingerRatioDbg("index");
    var rm = fingerRatioDbg("mid");
    var rr = fingerRatioDbg("ring");
    var rp = fingerRatioDbg("pinky");
    var rt = fingerRatioDbg("thumb");
    var ext = script.extendRatio;
    var iExt = ri >= ext;
    var mExt = rm >= ext;
    var rExt = rr >= ext;
    var pExt = rp >= ext;
    var nUp = (iExt ? 1 : 0) + (mExt ? 1 : 0) + (rExt ? 1 : 0) + (pExt ? 1 : 0);
    var curlMax = script.curlMaxRatio;
    var idxC = ri >= 0 && ri <= curlMax;
    var midC = rm >= 0 && rm <= curlMax;
    var rngC = rr >= 0 && rr <= curlMax;
    var pm = bendAmount("pinky-0", "pinky-1", "pinky-2");
    var pd = bendAmount("pinky-1", "pinky-2", "pinky-3");
    var pinkyStraight = pm >= 0 && pd >= 0 && pm <= 0.12 && pd <= 0.12;
    var pinkyOk = rp >= script.phonePinkyExtendRatio;
    var dTip = jdDbg("wrist", "thumb-3");
    var dMid = jdDbg("wrist", "thumb-1");
    var tHinge = dTip !== null && dMid !== null && dMid > 1e-5 && dTip / dMid >= script.thumbExtendRatio;
    var scale = jdDbg("wrist", "mid-0");
    if (scale === null || scale < 1e-5) {
        scale = 1;
    }
    var d0 = jdDbg("thumb-3", "index-0");
    var d1 = jdDbg("thumb-3", "index-1");
    var d2 = jdDbg("thumb-3", "index-2");
    var bestAlong = 999;
    if (d0 !== null) {
        bestAlong = Math.min(bestAlong, d0);
    }
    if (d1 !== null) {
        bestAlong = Math.min(bestAlong, d1);
    }
    if (d2 !== null) {
        bestAlong = Math.min(bestAlong, d2);
    }
    var alongN = bestAlong < 900 ? bestAlong / scale : -1;
    var maxAlongI = script.aslIThumbMaxAlongVersusY > 0 ? script.aslIThumbMaxAlongVersusY : 0.46;
    var minAlongY = script.aslYThumbMinAlongVersusI > 0 ? script.aslYThumbMinAlongVersusI : 0.48;
    var minTrY = script.aslYThumbMinWristRatio > 0 ? script.aslYThumbMinWristRatio : 2.0;
    var ratioAlongMin =
        script.aslYThumbRatioGateMinAlong > 0 ? script.aslYThumbRatioGateMinAlong : 0.36;
    var tCurl = rt >= 0 && rt <= curlMax;
    var yCore =
        pExt &&
        nUp === 1 &&
        pinkyOk &&
        pinkyStraight &&
        idxC &&
        midC &&
        rngC;
    var thumbOutY =
        (alongN >= 0 && alongN >= minAlongY) ||
        (rt >= 0 && rt >= minTrY && alongN > ratioAlongMin);
    var yThumbBand = tHinge && thumbOutY;
    var looksLikeI = yCore && !yThumbBand && (tCurl || !tHinge || (alongN >= 0 && alongN <= maxAlongI));
    print(
        "[Gesture][I debug] " +
            pfx +
            "maxAlongI=" +
            maxAlongI.toFixed(3) +
            " minAlongY(ref)=" +
            minAlongY.toFixed(3) +
            " minTrY(ref)=" +
            minTrY.toFixed(2) +
            " ratioAlongMin=" +
            ratioAlongMin.toFixed(3) +
            " phonePinkyExtendRatio=" +
            script.phonePinkyExtendRatio.toFixed(3) +
            " ratios i/m/r/p/t=" +
            (ri < 0 ? "?" : ri.toFixed(3)) +
            "/" +
            (rm < 0 ? "?" : rm.toFixed(3)) +
            "/" +
            (rr < 0 ? "?" : rr.toFixed(3)) +
            "/" +
            (rp < 0 ? "?" : rp.toFixed(3)) +
            "/" +
            (rt < 0 ? "?" : rt.toFixed(3)) +
            " nUp=" +
            nUp +
            " pinkyOk=" +
            pinkyOk +
            " pinkyStraight=" +
            pinkyStraight +
            " i/m/r curled=" +
            idxC +
            "/" +
            midC +
            "/" +
            rngC +
            " thumbHingeOpen=" +
            tHinge +
            " thumbAlongN=" +
            (alongN < 0 ? "?" : alongN.toFixed(3)) +
            " thumbOutY=" +
            thumbOutY +
            " yThumbBand=" +
            yThumbBand +
            " => looksLikeAslI=" +
            looksLikeI
    );
}

script.clearKnnDataset = function () {
    saveKnnDataset({ v: POSE_FEATURE_VERSION, samples: [] });
    print("[Gesture] kNN dataset cleared");
};

script.getKnnSampleCount = function () {
    return loadKnnDataset().samples.length;
};

script.buildPoseFeatureVectorForHand = function (hs) {
    return buildPoseFeatureVector(hs);
};

script.debugHNow = function (hs) {
    var h = hs;
    if (!h) {
        var tt = getWatchTargets();
        if (tt.length > 0) {
            h = tt[0].script;
        }
    }
    debugHForHand(h, "manual");
};

script.debugINow = function (hs) {
    var h = hs;
    if (!h) {
        var tt = getWatchTargets();
        if (tt.length > 0) {
            h = tt[0].script;
        }
    }
    debugIForHand(h, "manual");
};

function classifyForHand(handScript) {
    function jd(a, b) {
        try {
            if (!handScript || !handScript.getJointsDistance) {
                return null;
            }
            return handScript.getJointsDistance(a, b);
        } catch (e) {
            return null;
        }
    }

    function getWorldPos(jName) {
        try {
            if (!handScript || !handScript.getJoint) {
                return null;
            }
            var joint = handScript.getJoint(jName);
            if (!joint || joint.position === undefined || joint.position === null) {
                return null;
            }
            return joint.position;
        } catch (e) {
            return null;
        }
    }

    function fingerRatio(prefix) {
        var dTip = jd("wrist", prefix + "-3");
        var dBase = jd("wrist", prefix + "-0");
        if (dTip === null || dBase === null || dBase < 1e-5) {
            return -1;
        }
        return dTip / dBase;
    }

    function fingerExtended(prefix) {
        var r = fingerRatio(prefix);
        if (r < 0) {
            return false;
        }
        return r >= script.extendRatio;
    }

    function fingerExtendedForB(prefix) {
        var r = fingerRatio(prefix);
        if (r < 0) {
            return false;
        }
        var thr = script.aslBExtendRatio > 0 ? script.aslBExtendRatio : script.extendRatio;
        return r >= thr;
    }

    function fingerExtendedAtLeast(prefix, minRatio) {
        var r = fingerRatio(prefix);
        if (r < 0) {
            return false;
        }
        return r >= minRatio;
    }

    function fingerCurled(prefix) {
        var r = fingerRatio(prefix);
        if (r < 0) {
            return false;
        }
        return r <= script.curlMaxRatio;
    }

    function thumbExtended() {
        var dTip = jd("wrist", "thumb-3");
        var dMid = jd("wrist", "thumb-1");
        if (dTip === null || dMid === null || dMid < 1e-5) {
            return false;
        }
        return dTip / dMid >= script.thumbExtendRatio;
    }

    var scale = jd("wrist", "mid-0");
    if (scale === null || scale < 1e-5) {
        scale = 1;
    }

    function palmFrame() {
        try {
            var w = getWorldPos("wrist");
            var mid0 = getWorldPos("mid-0");
            var idx0 = getWorldPos("index-0");
            if (!w || !mid0 || !idx0) {
                return null;
            }
            var u = mid0.sub(w);
            var lenU = u.length;
            if (lenU < 1e-5) {
                return null;
            }
            u = u.uniformScale(1 / lenU);
            var v = idx0.sub(w);
            var vPerpU = v.sub(u.uniformScale(v.dot(u)));
            var lenV = vPerpU.length;
            if (lenV < 1e-5) {
                return null;
            }
            vPerpU = vPerpU.uniformScale(1 / lenV);
            var n = u.cross(vPerpU);
            var lenN = n.length;
            if (lenN < 1e-5) {
                return null;
            }
            n = n.uniformScale(1 / lenN);
            return { w: w, u: u, v: vPerpU, n: n };
        } catch (ePf) {
            return null;
        }
    }

    function thumbOutOfPalmNorm() {
        var pf = palmFrame();
        var t3 = getWorldPos("thumb-3");
        if (!pf || !t3) {
            return 0;
        }
        var outward = t3.sub(pf.w).dot(pf.n);
        return outward / scale;
    }

    function indexChainInPlaneSpreadNorm() {
        try {
            var pf = palmFrame();
            if (!pf) {
                return 999;
            }
            var names = ["index-0", "index-1", "index-2", "index-3"];
            var pts = [];
            for (var a = 0; a < names.length; a++) {
                var p = getWorldPos(names[a]);
                if (!p) {
                    return 999;
                }
                var rel = p.sub(pf.w);
                var x = rel.dot(pf.u);
                var y = rel.dot(pf.v);
                pts.push({ x: x, y: y });
            }
            var x0 = pts[0].x;
            var y0 = pts[0].y;
            var x3 = pts[3].x;
            var y3 = pts[3].y;
            var dx = x3 - x0;
            var dy = y3 - y0;
            var baseLen = Math.sqrt(dx * dx + dy * dy);
            if (baseLen < 1e-5) {
                return 0;
            }
            var nx = -dy / baseLen;
            var ny = dx / baseLen;
            var maxPerp = 0;
            for (var b = 0; b < pts.length; b++) {
                var px = pts[b].x - x0;
                var py = pts[b].y - y0;
                var perp = Math.abs(px * nx + py * ny);
                if (perp > maxPerp) {
                    maxPerp = perp;
                }
            }
            return maxPerp / scale;
        } catch (e) {
            return 999;
        }
    }

    function peaceSpreadOk() {
        var spread = jd("index-3", "mid-3");
        var knuckle = jd("index-0", "mid-0");
        if (spread === null || knuckle === null || knuckle < 1e-5) {
            return true;
        }
        return spread >= knuckle * script.peaceSpreadRatio;
    }

    function peaceCompanionFingerDown(prefix) {
        var ratio = fingerRatio(prefix);
        if (ratio < 0) {
            return false;
        }
        if (ratio >= script.extendRatio) {
            return false;
        }
        return ratio <= script.peaceCompanionMaxRatio;
    }

    var t = thumbExtended();
    var i = fingerExtended("index");
    var m = fingerExtended("mid");
    var r = fingerExtended("ring");
    var p = fingerExtended("pinky");

    var iPeace = fingerExtendedAtLeast("index", script.peaceFingerExtendRatio);
    var mPeace = fingerExtendedAtLeast("mid", script.peaceFingerExtendRatio);

    var spreadIJ = jd("index-3", "mid-3");
    var knuckleIJ = jd("index-0", "mid-0");
    var tipToKnuckle = -1;
    if (spreadIJ !== null && knuckleIJ !== null && knuckleIJ > 1e-5) {
        tipToKnuckle = spreadIJ / knuckleIJ;
    }

    var nFingersStrict = (i ? 1 : 0) + (m ? 1 : 0) + (r ? 1 : 0) + (p ? 1 : 0);

    var allFourStrict = i && m && r && p;
    var allFourForB = fingerExtendedForB("index") && fingerExtendedForB("mid") && fingerExtendedForB("ring") && fingerExtendedForB("pinky");
    var allFourCurled =
        fingerCurled("index") && fingerCurled("mid") && fingerCurled("ring") && fingerCurled("pinky");

    function fingerCurledForAslA(prefix) {
        var ratio = fingerRatio(prefix);
        if (ratio < 0) {
            return false;
        }
        return ratio <= script.aslACurlMaxRatio;
    }

    var allFourCurledForAslA =
        fingerCurledForAslA("index") &&
        fingerCurledForAslA("mid") &&
        fingerCurledForAslA("ring") &&
        fingerCurledForAslA("pinky");

    function bendAmountAt(a, b, c) {
        var pa = getWorldPos(a);
        var pb = getWorldPos(b);
        var pc = getWorldPos(c);
        if (!pa || !pb || !pc) {
            return -1;
        }
        var v1 = pa.sub(pb);
        var v2 = pc.sub(pb);
        var l1 = v1.length;
        var l2 = v2.length;
        if (l1 < 1e-5 || l2 < 1e-5) {
            return -1;
        }
        v1 = v1.uniformScale(1 / l1);
        v2 = v2.uniformScale(1 / l2);
        var d = v1.dot(v2);
        if (d > 1) {
            d = 1;
        }
        if (d < -1) {
            d = -1;
        }
        var ang = Math.acos(d);
        return (Math.PI - ang) / Math.PI; // 0 straight, 1 curled
    }

    function fingerMiddleAndDistalCurled(prefix) {
        var midCurl = bendAmountAt(prefix + "-0", prefix + "-1", prefix + "-2");
        var distCurl = bendAmountAt(prefix + "-1", prefix + "-2", prefix + "-3");
        if (midCurl < 0 || distCurl < 0) {
            return false;
        }
        return midCurl >= 0.16 && distCurl >= 0.16;
    }

    function fingerMiddleAndDistalStraight(prefix) {
        var midCurl = bendAmountAt(prefix + "-0", prefix + "-1", prefix + "-2");
        var distCurl = bendAmountAt(prefix + "-1", prefix + "-2", prefix + "-3");
        if (midCurl < 0 || distCurl < 0) {
            return false;
        }
        return midCurl <= 0.12 && distCurl <= 0.12;
    }

    function thumbAlongIndexNorm() {
        if (scale < 1e-5) {
            return 999;
        }
        var d0 = jd("thumb-3", "index-0");
        var d1 = jd("thumb-3", "index-1");
        var d2 = jd("thumb-3", "index-2");
        var best = 999;
        if (d0 !== null) {
            best = Math.min(best, d0);
        }
        if (d1 !== null) {
            best = Math.min(best, d1);
        }
        if (d2 !== null) {
            best = Math.min(best, d2);
        }
        return best / scale;
    }

    /** Averaged index+mid chain direction vs palm |normal|: used with separate H/U thresholds (lens-space; tune inputs). */
    function pairIndexMidAbsDotPalmN() {
        var pf = palmFrame();
        if (!pf) {
            return -1;
        }
        var i0 = getWorldPos("index-0");
        var i3 = getWorldPos("index-3");
        var m0 = getWorldPos("mid-0");
        var m3 = getWorldPos("mid-3");
        if (!i0 || !i3 || !m0 || !m3) {
            return -1;
        }
        var di = i3.sub(i0);
        var dm = m3.sub(m0);
        var dir = di.add(dm);
        var len = dir.length;
        if (len < 1e-5) {
            return -1;
        }
        dir = dir.uniformScale(1 / len);
        return Math.abs(dir.dot(pf.n));
    }

    function pairIndexMidWorldVerticalness() {
        var i0 = getWorldPos("index-0");
        var i3 = getWorldPos("index-3");
        var m0 = getWorldPos("mid-0");
        var m3 = getWorldPos("mid-3");
        if (!i0 || !i3 || !m0 || !m3) {
            return -1;
        }
        var di = i3.sub(i0);
        var dm = m3.sub(m0);
        var dir = di.add(dm);
        var len = dir.length;
        if (len < 1e-5) {
            return -1;
        }
        dir = dir.uniformScale(1 / len);
        var up = new vec3(0, 1, 0);
        return Math.abs(dir.dot(up));
    }

    function looksLikeAslLetterA() {
        if (!allFourCurledForAslA) {
            return false;
        }
        if (thumbOutOfPalmNorm() > script.aslAThumbOutOfPalmMax) {
            return false;
        }
        var inPlaneSpread = indexChainInPlaneSpreadNorm();
        if (inPlaneSpread > 0.55) {
            return false;
        }
        var pn = thumbAlongIndexNorm();
        var hard = Math.max(script.aslAThumbIndexMaxRatio, 0.84);
        var soft = hard + script.aslAThumbIndexSoftMargin;
        if (pn > soft) {
            return false;
        }
        if (pn > hard && t) {
            return false;
        }
        if (!script.aslAThumbNearerIndexThanRing) {
            return true;
        }
        var dTI = jd("thumb-3", "index-1");
        var dTR = jd("thumb-3", "ring-1");
        if (dTI === null || dTR === null || dTR < 1e-5) {
            return true;
        }
        return dTI < dTR * 0.95;
    }

    var aslLetterA = looksLikeAslLetterA();

    function looksLikeAslO() {
        if (!t || !fingerCurled("index") || !fingerCurled("mid") || !fingerCurled("ring") || !fingerCurled("pinky")) {
            return false;
        }
        var tips = ["index-3", "mid-3", "ring-3", "pinky-3"];
        var sum = 0;
        var cnt = 0;
        for (var c = 0; c < tips.length; c++) {
            var d = jd("thumb-3", tips[c]);
            if (d !== null) {
                sum += d;
                cnt++;
            }
        }
        if (cnt < 1) {
            return false;
        }
        return sum / cnt / scale <= script.aslOMaxAvgTipDist;
    }

    function looksLikeAslR() {
        if (!iPeace || !mPeace || !peaceCompanionFingerDown("ring") || !peaceCompanionFingerDown("pinky")) {
            return false;
        }
        if (tipToKnuckle < 0) {
            return false;
        }
        return tipToKnuckle < script.aslRMaxTipToKnuckle;
    }

    function looksLikeAslH() {
        // H: index + middle extended; ring, pinky curled; thumb tucked (ratio can lie — use along-index).
        // Wrist-relative: pair direction mostly in palm plane (not vertical like U).
        if (!i || !m || nFingersStrict !== 2) {
            return false;
        }
        if (!fingerCurled("ring") || !fingerCurled("pinky")) {
            return false;
        }
        var palmN = pairIndexMidAbsDotPalmN();
        var minPalmN = script.aslHPairMinPalmN > 0 ? script.aslHPairMinPalmN : 0.46;
        if (palmN < 0 || palmN < minPalmN) {
            return false;
        }
        var along = thumbAlongIndexNorm();
        var maxAlong = script.aslHThumbMaxAlongIndexNorm > 0 ? script.aslHThumbMaxAlongIndexNorm : 1.18;
        var thumbTucked = fingerCurled("thumb") || along <= maxAlong;
        if (!thumbTucked) {
            return false;
        }
        return true;
    }

    function looksLikeAslU() {
        if (!iPeace || !mPeace || !peaceCompanionFingerDown("ring") || !peaceCompanionFingerDown("pinky")) {
            return false;
        }
        if (tipToKnuckle < 0) {
            return false;
        }
        if (!(tipToKnuckle > script.aslHMaxTipToKnuckle && tipToKnuckle <= script.aslUMaxTipToKnuckle)) {
            return false;
        }
        var palmN = pairIndexMidAbsDotPalmN();
        var maxPalmN = script.aslUPairMaxPalmN > 0 ? script.aslUPairMaxPalmN : 0.4;
        if (palmN < 0 || palmN > maxPalmN) {
            return false;
        }
        return true;
    }

    function looksLikeAslV() {
        // V should be strict two-finger extension (avoid 1-finger poses being misread as V
        // when peaceFingerExtendRatio is tuned low).
        if (!i || !m || nFingersStrict !== 2) {
            return false;
        }
        return (
            peaceCompanionFingerDown("ring") &&
            peaceCompanionFingerDown("pinky") &&
            peaceSpreadOk() &&
            tipToKnuckle > script.aslUMaxTipToKnuckle
        );
    }

    function yPinkyCoreForIY() {
        var pinkyR = fingerRatio("pinky");
        return (
            p &&
            nFingersStrict === 1 &&
            pinkyR >= script.phonePinkyExtendRatio &&
            fingerMiddleAndDistalStraight("pinky") &&
            fingerCurled("index") &&
            fingerCurled("mid") &&
            fingerCurled("ring")
        );
    }

    /** True when thumb reads “out” for Y: far along index, or long wrist ratio only when along isn’t tucked (I can have high ratio + low along). */
    function yThumbOutVersusI() {
        var minAlongY = script.aslYThumbMinAlongVersusI > 0 ? script.aslYThumbMinAlongVersusI : 0.48;
        var minTrY = script.aslYThumbMinWristRatio > 0 ? script.aslYThumbMinWristRatio : 2.0;
        var ratioAlongMin =
            script.aslYThumbRatioGateMinAlong > 0 ? script.aslYThumbRatioGateMinAlong : 0.36;
        var along = thumbAlongIndexNorm();
        var tr = fingerRatio("thumb");
        var byAlong = along >= 0 && along >= minAlongY;
        var byRatio = tr >= 0 && tr >= minTrY && along > ratioAlongMin;
        return byAlong || byRatio;
    }

    function looksLikeAslY() {
        // Y: pinky core + hinge thumb extended + thumb out (along and/or wrist thumb ratio).
        return yPinkyCoreForIY() && t && yThumbOutVersusI();
    }

    function looksLikeAslW() {
        return i && m && r && !p && fingerCurled("pinky");
    }

    function indexTipBelowWristWorld() {
        var wi = getWorldPos("wrist");
        var i3 = getWorldPos("index-3");
        if (!wi || !i3) {
            return false;
        }
        return i3.y < wi.y;
    }

    function looksLikeAslF() {
        // F: like B but index curled toward thumb; mid/ring/pinky extended; thumb+index pinch (O).
        var pinch = jd("thumb-3", "index-3");
        if (pinch === null || scale < 1e-5) {
            return false;
        }
        var pinchN = pinch / scale;
        if (pinchN > 0.52) {
            return false;
        }
        if (!fingerExtendedForB("mid") || !fingerExtendedForB("ring") || !fingerExtendedForB("pinky")) {
            return false;
        }
        if (!fingerMiddleAndDistalCurled("index")) {
            return false;
        }
        return true;
    }

    function looksLikeGShape() {
        // G: index extended; middle/ring/pinky curled; thumb up and close to index.
        if (!i) {
            return false;
        }
        if (!fingerCurled("mid") || !fingerCurled("ring") || !fingerCurled("pinky")) {
            return false;
        }
        if (!t) {
            return false;
        }
        var thumbTip = getWorldPos("thumb-3");
        var thumbMid = getWorldPos("thumb-1");
        if (!thumbTip || !thumbMid) {
            return false;
        }
        if (thumbTip.y <= thumbMid.y) {
            return false;
        }
        var dTip = jd("thumb-3", "index-3");
        var dBase = jd("thumb-3", "index-1");
        if (dTip === null || dBase === null || scale < 1e-5) {
            return false;
        }
        var nearTipN = dTip / scale;
        var nearBaseN = dBase / scale;
        return nearTipN <= 0.95 && nearBaseN <= 0.95;
    }

    function looksLikeAslQ() {
        return looksLikeGShape() && indexTipBelowWristWorld();
    }

    function looksLikeAslG() {
        // User intent: G depends on finger/thumb shape, not wrist-relative direction.
        return looksLikeGShape();
    }

    function looksLikeTwoFingerWithThumbOnMid() {
        // K/P should require strict index + middle extension, not loose peace thresholds.
        if (!i || !m || nFingersStrict !== 2) {
            return false;
        }
        if (!peaceCompanionFingerDown("ring") || !peaceCompanionFingerDown("pinky")) {
            return false;
        }
        var dTM = jd("thumb-3", "mid-1");
        if (dTM === null || scale < 1e-5) {
            return false;
        }
        if (dTM / scale > 0.56) {
            return false;
        }
        if (tipToKnuckle < 0 || tipToKnuckle <= script.aslHMaxTipToKnuckle) {
            return false;
        }
        // Reject near-parallel U-like hand where fingers are too close together.
        if (tipToKnuckle <= script.aslUMaxTipToKnuckle) {
            return false;
        }
        return true;
    }

    function looksLikeAslP() {
        return looksLikeTwoFingerWithThumbOnMid() && indexTipBelowWristWorld();
    }

    function looksLikeAslK() {
        return looksLikeTwoFingerWithThumbOnMid() && !indexTipBelowWristWorld();
    }

    function looksLikeAslC() {
        // C: open-ish curved hand with medium thumb-index gap (not pinch, not flat open B).
        var ri = fingerRatio("index");
        var rm = fingerRatio("mid");
        var rr = fingerRatio("ring");
        var rp = fingerRatio("pinky");
        var minForC = Math.max(script.peaceFingerExtendRatio, script.aslBExtendRatio * 0.92);
        if (ri < minForC || rm < minForC || rr < minForC || rp < minForC) {
            return false;
        }
        // Avoid B being consumed as C: if all four fingers are very straight, prefer B.
        var cStraight = script.aslCMaxStraightRatio > 0 ? script.aslCMaxStraightRatio : 1.55;
        if (ri > cStraight && rm > cStraight && rr > cStraight && rp > cStraight) {
            return false;
        }
        var tr0 = fingerRatio("thumb");
        if (tr0 >= 0 && tr0 < script.extendRatio * 0.7) {
            return false;
        }
        var dTI = jd("thumb-3", "index-3");
        if (dTI === null || scale < 1e-5) {
            return false;
        }
        var tiN = dTI / scale;
        if (tiN <= 0.30 || tiN >= 1.35) {
            return false;
        }
        var dTM = jd("thumb-3", "mid-3");
        var dTR = jd("thumb-3", "ring-3");
        var dTP = jd("thumb-3", "pinky-3");
        var sum = 0;
        var cnt = 0;
        if (dTM !== null) {
            sum += dTM / scale;
            cnt++;
        }
        if (dTR !== null) {
            sum += dTR / scale;
            cnt++;
        }
        if (dTP !== null) {
            sum += dTP / scale;
            cnt++;
        }
        if (cnt < 1) {
            return false;
        }
        var avgOther = sum / cnt;
        if (avgOther < 0.35 || avgOther > 1.8) {
            return false;
        }
        // Exclude F-like pinch.
        if (tiN < 0.44) {
            return false;
        }
        return true;
    }

    function looksLikeAslT() {
        if (nFingersStrict !== 0 || t || aslLetterA) {
            return false;
        }
        if (!allFourCurled) {
            return false;
        }
        var d = jd("thumb-3", "index-1");
        if (d === null || scale < 1e-5) {
            return false;
        }
        if (d / scale >= 0.45) {
            return false;
        }
        if (thumbAlongIndexNorm() > 0.85) {
            return false;
        }
        return true;
    }

    function looksLikeAslE() {
        if (nFingersStrict !== 0 || aslLetterA) {
            return false;
        }
        if (
            !fingerMiddleAndDistalCurled("index") ||
            !fingerMiddleAndDistalCurled("mid") ||
            !fingerMiddleAndDistalCurled("ring") ||
            !fingerMiddleAndDistalCurled("pinky")
        ) {
            return false;
        }
        var pf = palmFrame();
        var t3 = getWorldPos("thumb-3");
        if (!pf || !t3) {
            return false;
        }
        var rel = t3.sub(pf.w);
        var tu = rel.dot(pf.u) / scale;
        var tv = rel.dot(pf.v) / scale;
        var tn = rel.dot(pf.n) / scale;
        var maxAbsU = script.aslEThumbMaxAbsU > 0 ? script.aslEThumbMaxAbsU : 1.05;
        var maxAbsV = script.aslEThumbMaxAbsV > 0 ? script.aslEThumbMaxAbsV : 0.65;
        var minN = Math.min(script.aslEThumbMinN, -0.45);
        var maxN = script.aslEThumbMaxN;
        if (Math.abs(tu) > maxAbsU || Math.abs(tv) > maxAbsV || tn < minN || tn > maxN) {
            return false;
        }
        if (looksLikeAslS()) {
            return false;
        }
        if (looksLikeAslL()) {
            return false;
        }
        return true;
    }

    function looksLikeAslM() {
        if (nFingersStrict !== 0 || t || aslLetterA) {
            return false;
        }
        if (!allFourCurled) {
            return false;
        }
        if (looksLikeAslS() || looksLikeAslT() || looksLikeAslE()) {
            return false;
        }
        var dR = jd("thumb-3", "ring-1");
        var dI = jd("thumb-3", "index-1");
        if (dR === null || dI === null || scale < 1e-5 || dI < 1e-5) {
            return false;
        }
        return dR / scale < dI / scale * 0.9;
    }

    function looksLikeAslN() {
        if (nFingersStrict !== 0 || t || aslLetterA) {
            return false;
        }
        if (!allFourCurled) {
            return false;
        }
        if (looksLikeAslS() || looksLikeAslT() || looksLikeAslE() || looksLikeAslM()) {
            return false;
        }
        var dM = jd("thumb-3", "mid-1");
        var dI = jd("thumb-3", "index-1");
        if (dM === null || dI === null || scale < 1e-5 || dI < 1e-5) {
            return false;
        }
        return dM / scale < dI / scale * 0.88 && dM / scale < 0.48;
    }

    function thumbHingeOpenForAslL() {
        var tr = fingerRatio("thumb");
        if (tr < 0) {
            return false;
        }
        if (fingerCurled("thumb")) {
            return false;
        }
        return tr >= script.extendRatio * 0.88;
    }

    function looksLikeAslL() {
        // User-tuned: L only when thumb + index are extended and other fingers are curled.
        return (
            t &&
            i &&
            fingerCurled("mid") &&
            fingerCurled("ring") &&
            fingerCurled("pinky")
        );
    }

    function looksLikeAslD() {
        // D: index up, others down. Thumb should not form L, but can be slightly open.
        // Be tolerant to index not being fully straight.
        var ri0 = fingerRatio("index");
        var minIndexForD = Math.max(script.peaceFingerExtendRatio, script.extendRatio * 0.9);
        if (ri0 < 0 || ri0 < minIndexForD) {
            return false;
        }
        var midR = fingerRatio("mid");
        var ringR = fingerRatio("ring");
        var pinkyR = fingerRatio("pinky");
        if (midR >= 0 && midR >= script.extendRatio * 0.92) {
            return false;
        }
        if (ringR >= 0 && ringR >= script.extendRatio * 0.92) {
            return false;
        }
        if (pinkyR >= 0 && pinkyR >= script.extendRatio * 0.92) {
            return false;
        }
        // Only reject clear L-like thumb separation, not merely a high thumb ratio.
        var thumbAlong = thumbAlongIndexNorm();
        if (thumbHingeOpenForAslL() && thumbAlong > script.aslAThumbIndexMaxRatio + 0.18) {
            return false;
        }
        var tr = fingerRatio("thumb");
        if (tr < 0) {
            return true;
        }
        return fingerCurled("thumb") || thumbAlong <= script.aslAThumbIndexMaxRatio + 0.22;
    }

    function looksLikeAslI() {
        // I: same pinky layout as Y, but not when thumb reads as Y-out (along or high wrist thumb ratio with t).
        var maxAlongI = script.aslIThumbMaxAlongVersusY > 0 ? script.aslIThumbMaxAlongVersusY : 0.46;
        var along = thumbAlongIndexNorm();
        if (!yPinkyCoreForIY()) {
            return false;
        }
        if (t && yThumbOutVersusI()) {
            return false;
        }
        return fingerCurled("thumb") || !t || along <= maxAlongI;
    }

    function looksLikeAslB() {
        // B: all four non-thumb fingers extended from palm.
        // Thumb is ignored (per spec): only index/mid/ring/pinky matter.
        // Require the four fingers to be straight (avoid E-like curled fingers still having high ratios).
        return (
            allFourForB &&
            fingerMiddleAndDistalStraight("index") &&
            fingerMiddleAndDistalStraight("mid") &&
            fingerMiddleAndDistalStraight("ring") &&
            fingerMiddleAndDistalStraight("pinky")
        );
    }

    function looksLikeAslX() {
        var ir = fingerRatio("index");
        if (ir < 0) {
            return false;
        }
        if (ir < 0.88 || ir >= script.extendRatio) {
            return false;
        }
        return fingerCurled("mid") && fingerCurled("ring") && fingerCurled("pinky") && !t;
    }

    function looksLikeAslS() {
        if (!allFourCurled || t || aslLetterA) {
            return false;
        }
        if (nFingersStrict !== 0) {
            return false;
        }
        return thumbAlongIndexNorm() > script.aslAThumbIndexMaxRatio + 0.08;
    }

    function safeRatio(prefix) {
        var v = fingerRatio(prefix);
        return v >= 0 ? v : 1.25;
    }

    function bandScore(value, lo, hi, peak) {
        if (value < lo) {
            return peak * Math.max(0, 1 - (lo - value) / (hi - lo + 0.01));
        }
        if (value > hi) {
            return peak * Math.max(0, 1 - (value - hi) / (hi - lo + 0.01));
        }
        return peak;
    }

    function avgThumbToFingerTipsNorm() {
        var tips = ["index-3", "mid-3", "ring-3", "pinky-3"];
        var sum = 0;
        var cnt = 0;
        for (var at = 0; at < tips.length; at++) {
            var d = jd("thumb-3", tips[at]);
            if (d !== null) {
                sum += d;
                cnt++;
            }
        }
        if (cnt < 1 || scale < 1e-5) {
            return 2.5;
        }
        return sum / cnt / scale;
    }

    function bestGuessLetter() {
        var ri = safeRatio("index");
        var rm = safeRatio("mid");
        var rr = safeRatio("ring");
        var rp = safeRatio("pinky");
        var rt = safeRatio("thumb");
        var tt = t ? 1 : 0;
        var tkn = tipToKnuckle >= 0 ? tipToKnuckle : 0.55;
        var ta = thumbAlongIndexNorm();
        if (ta > 6) {
            ta = 6;
        }
        var tout = thumbOutOfPalmNorm();
        if (tout > 2) {
            tout = 2;
        }
        if (tout < -1) {
            tout = -1;
        }
        var tipRing = avgThumbToFingerTipsNorm();
        var spreadOk = peaceSpreadOk() ? 1 : 0;
        var nUp = nFingersStrict;

        var S = {};

        S.fist =
            bandScore(ri, 0.85, 1.22, 4) +
            bandScore(rm, 0.85, 1.22, 4) +
            bandScore(rr, 0.85, 1.22, 4) +
            bandScore(rp, 0.85, 1.22, 4) +
            (1 - tt) * 5 +
            bandScore(rt, 0.9, 1.35, 2);

        S.asl_a =
            bandScore(ri, 0.92, 1.22, 4) +
            bandScore(rm, 0.92, 1.22, 4) +
            bandScore(rr, 0.92, 1.22, 4) +
            bandScore(rp, 0.92, 1.22, 4) +
            bandScore(ta, 0.2, 0.88, 8) +
            bandScore(tout, -0.2, 0.45, 5) +
            (tipRing > 0.55 ? 0 : 2);

        S.asl_o =
            bandScore(ri, 0.88, 1.2, 3) +
            bandScore(rm, 0.88, 1.2, 3) +
            bandScore(rr, 0.88, 1.2, 3) +
            bandScore(rp, 0.88, 1.2, 3) +
            tt * 4 +
            bandScore(rt, 1.05, 2.2, 4) +
            bandScore(tipRing, 0.25, 0.95, 10) +
            (ta < 0.95 ? 0 : -4);

        S.thumbs_up =
            tt * 6 +
            bandScore(ri, 0.88, 1.2, 3) +
            bandScore(rm, 0.88, 1.2, 3) +
            bandScore(rr, 0.88, 1.2, 3) +
            bandScore(rp, 0.88, 1.2, 3) +
            bandScore(Math.max(tout, ta * 0.35), 0.35, 2.5, 8) +
            (aslLetterA ? -15 : 0);

        S.asl_y =
            tt * 5 +
            bandScore(rp, 1.05, 2.4, 8) +
            bandScore(ri, 0.85, 1.2, 3) +
            bandScore(rm, 0.85, 1.2, 3) +
            bandScore(rr, 0.85, 1.2, 3);

        S.asl_w =
            bandScore(ri, 1.05, 2.2, 6) +
            bandScore(rm, 1.05, 2.2, 6) +
            bandScore(rr, 1.05, 2.2, 6) +
            bandScore(rp, 0.75, 1.18, 5);

        S.asl_v =
            bandScore(ri, 1.02, 2.2, 6) +
            bandScore(rm, 1.02, 2.2, 6) +
            bandScore(rr, 0.75, 1.2, 3) +
            bandScore(rp, 0.75, 1.2, 3) +
            bandScore(tkn, 0.95, 2.6, 10) +
            spreadOk * 3 +
            (iPeace && mPeace ? 5 : 0);

        S.asl_h =
            bandScore(ri, 1.0, 2.0, 5) +
            bandScore(rm, 1.0, 2.0, 5) +
            bandScore(rr, 0.75, 1.2, 3) +
            bandScore(rp, 0.75, 1.2, 3) +
            bandScore(tkn, 0.28, 0.62, 10);

        S.asl_u =
            bandScore(ri, 1.0, 2.0, 5) +
            bandScore(rm, 1.0, 2.0, 5) +
            bandScore(rr, 0.75, 1.2, 3) +
            bandScore(rp, 0.75, 1.2, 3) +
            bandScore(tkn, 0.58, 1.05, 10);

        S.asl_r =
            bandScore(ri, 1.0, 2.0, 5) +
            bandScore(rm, 1.0, 2.0, 5) +
            bandScore(rr, 0.75, 1.2, 3) +
            bandScore(rp, 0.75, 1.2, 3) +
            bandScore(tkn, 0.05, 0.34, 12);

        S.asl_l =
            tt * 5 +
            bandScore(ri, 1.05, 2.3, 8) +
            bandScore(rm, 0.75, 1.15, 3) +
            bandScore(rr, 0.75, 1.15, 3) +
            bandScore(rp, 0.75, 1.15, 3) +
            bandScore(rt, script.extendRatio * 0.84, script.extendRatio * 1.32, 6);

        S.asl_i =
            bandScore(rp, 1.05, 2.4, 10) +
            (1 - tt) * 2 +
            bandScore(ri, 0.75, 1.15, 3) +
            bandScore(rm, 0.75, 1.15, 3) +
            bandScore(rr, 0.75, 1.15, 3);

        var midPeaceHint = fingerExtendedAtLeast("mid", script.peaceFingerExtendRatio * 0.92) ? 1 : 0;
        var thumbOpenHint = rt >= script.extendRatio * 0.86 && !fingerCurled("thumb") ? 1 : 0;

        S.asl_d =
            bandScore(ri, 1.08, 2.4, 10) +
            bandScore(rm, 0.75, 1.12, 4) +
            bandScore(rr, 0.75, 1.12, 4) +
            bandScore(rp, 0.75, 1.12, 4) +
            (nUp === 1 ? 4 : 0) -
            midPeaceHint * 14 -
            (iPeace && mPeace ? 12 : 0) -
            tt * 10 -
            thumbOpenHint * 9 -
            bandScore(rm, 1.0, 2.15, 6);

        S.asl_b =
            bandScore(ri, 1.12, 2.3, 6) +
            bandScore(rm, 1.12, 2.3, 6) +
            bandScore(rr, 1.12, 2.3, 6) +
            bandScore(rp, 1.12, 2.3, 6) +
            (1 - tt) * 4 +
            bandScore(rt, 0.85, 1.25, 4);

        S.asl_x =
            bandScore(ri, 0.9, 1.12, 8) +
            bandScore(rm, 0.75, 1.15, 3) +
            bandScore(rr, 0.75, 1.15, 3) +
            bandScore(rp, 0.75, 1.15, 3) +
            (1 - tt) * 2;

        S.asl_s =
            bandScore(ri, 0.88, 1.22, 4) +
            bandScore(rm, 0.88, 1.22, 4) +
            bandScore(rr, 0.88, 1.22, 4) +
            bandScore(rp, 0.88, 1.22, 4) +
            (1 - tt) * 4 +
            bandScore(ta, 0.72, 2.5, 8) +
            (aslLetterA ? -10 : 0);

        var pinchTI =
            jd("thumb-3", "index-3") !== null && scale > 1e-5 ? jd("thumb-3", "index-3") / scale : 1.2;
        var dTMid =
            jd("thumb-3", "mid-1") !== null && scale > 1e-5 ? jd("thumb-3", "mid-1") / scale : 1.2;

        S.asl_c =
            bandScore(ri, 1.05, 2.25, 5) +
            bandScore(rm, 1.05, 2.25, 5) +
            bandScore(rr, 1.05, 2.25, 5) +
            bandScore(rp, 1.05, 2.25, 5) +
            bandScore(rt, script.extendRatio * 0.9, script.extendRatio * 1.35, 6) +
            (1 - tt) * 2;

        S.asl_e =
            bandScore(ri, 0.75, 1.12, 5) +
            bandScore(rm, 0.75, 1.12, 5) +
            bandScore(rr, 0.75, 1.12, 5) +
            bandScore(rp, 0.75, 1.12, 5) +
            bandScore(rt, 0.75, 1.12, 5) +
            (1 - tt) * 4 +
            bandScore(ta, 0.15, 0.78, 6) +
            bandScore(tipRing, 0.2, 0.65, 8);

        S.asl_f =
            bandScore(rm, 1.05, 2.35, 7) +
            bandScore(rr, 1.05, 2.35, 7) +
            bandScore(rp, 1.05, 2.35, 7) +
            bandScore(ri, 0.72, 1.05, 5) +
            bandScore(pinchTI, 0.08, 0.42, 12);

        S.asl_g =
            bandScore(ri, 1.05, 2.35, 7) +
            bandScore(rm, 0.75, 1.15, 4) +
            bandScore(rr, 0.75, 1.15, 4) +
            bandScore(rp, 0.75, 1.15, 4) +
            bandScore(pinchTI, 0.22, 1.05, 8) +
            bandScore(rt, script.extendRatio * 0.8, script.extendRatio * 1.35, 5);

        S.asl_j = S.asl_i + bandScore(rp, 1.08, 2.5, 2) * 0.15;

        S.asl_k =
            bandScore(ri, 1.02, 2.2, 5) +
            bandScore(rm, 1.02, 2.2, 5) +
            bandScore(rr, 0.75, 1.2, 3) +
            bandScore(rp, 0.75, 1.2, 3) +
            bandScore(tkn, 0.95, 2.6, 6) +
            bandScore(dTMid, 0.12, 0.48, 12) +
            spreadOk * 2 +
            (nUp === 2 ? 4 : -9) +
            (rm >= script.extendRatio ? 3 : -10) +
            (ri >= script.extendRatio ? 2 : -6);

        S.asl_m =
            bandScore(ri, 0.82, 1.18, 4) +
            bandScore(rm, 0.82, 1.18, 4) +
            bandScore(rr, 0.82, 1.18, 4) +
            bandScore(rp, 0.82, 1.18, 4) +
            (1 - tt) * 4 +
            bandScore(ta, 0.25, 0.72, 7);

        S.asl_n =
            bandScore(ri, 0.82, 1.18, 4) +
            bandScore(rm, 0.82, 1.18, 4) +
            bandScore(rr, 0.82, 1.18, 4) +
            bandScore(rp, 0.82, 1.18, 4) +
            (1 - tt) * 4 +
            bandScore(ta, 0.28, 0.78, 8);

        S.asl_p = S.asl_k + (indexTipBelowWristWorld() ? 4 : -4);

        S.asl_q = S.asl_g + (indexTipBelowWristWorld() ? 5 : -3);

        S.asl_t =
            bandScore(ri, 0.82, 1.18, 4) +
            bandScore(rm, 0.82, 1.18, 4) +
            bandScore(rr, 0.82, 1.18, 4) +
            bandScore(rp, 0.82, 1.18, 4) +
            (1 - tt) * 4 +
            bandScore(ta, 0.2, 0.55, 9);

        S.asl_z =
            bandScore(ri, 1.05, 2.35, 5) +
            bandScore(rm, 0.75, 1.18, 3) +
            bandScore(rr, 0.75, 1.15, 3) +
            bandScore(rp, 0.75, 1.15, 3) +
            (1 - tt) * 2 +
            bandScore(tout, 0.1, 0.55, 4);

        var bestKey = "asl_v";
        var bestVal = -1e9;
        for (var key in S) {
            if (S[key] > bestVal) {
                bestVal = S[key];
                bestKey = key;
            }
        }
        return bestKey;
    }

    if (looksLikeAslR()) {
        return "asl_r";
    }
    if (looksLikeAslE()) {
        return "asl_e";
    }
    if (looksLikeAslG()) {
        return "asl_g";
    }
    if (looksLikeAslD()) {
        return "asl_d";
    }
    if (looksLikeAslP()) {
        return "asl_p";
    }
    if (looksLikeAslK()) {
        return "asl_k";
    }
    if (looksLikeAslY()) {
        return "asl_y";
    }
    if (looksLikeAslI()) {
        return "asl_i";
    }
    if (looksLikeAslF()) {
        return "asl_f";
    }
    if (looksLikeAslW()) {
        return "asl_w";
    }
    if (looksLikeAslH()) {
        return "asl_h";
    }
    if (looksLikeAslU()) {
        return "asl_u";
    }
    if (looksLikeAslV()) {
        return "asl_v";
    }
    if (looksLikeAslQ()) {
        return "asl_q";
    }
    if (looksLikeAslL()) {
        return "asl_l";
    }
    if (looksLikeAslC()) {
        return "asl_c";
    }
    if (looksLikeAslB()) {
        return "asl_b";
    }
    if (aslLetterA) {
        return "asl_a";
    }
    if (looksLikeAslO()) {
        return "asl_o";
    }
    if (
        t &&
        allFourCurled &&
        !aslLetterA &&
        (thumbOutOfPalmNorm() >= script.aslThumbsUpMinOutOfPalm ||
            thumbAlongIndexNorm() > script.aslAThumbIndexMaxRatio + 0.08)
    ) {
        return "thumbs_up";
    }
    if (looksLikeAslX()) {
        return "asl_x";
    }
    if (looksLikeAslS()) {
        return "asl_s";
    }
    if (looksLikeAslT()) {
        return "asl_t";
    }
    if (looksLikeAslN()) {
        return "asl_n";
    }
    if (looksLikeAslM()) {
        return "asl_m";
    }
    if (!t && nFingersStrict === 0) {
        return "fist";
    }
    if (allFourForB) {
        return "asl_b";
    }
    return bestGuessLetter();
}

function processTarget(target, precomputedLabel) {
    var label = precomputedLabel;
    if (label === undefined || label === null) {
        label = "unknown";
        try {
            label = classifyForHand(target.script);
        } catch (err) {
            print("[Gesture] classify error (check HandTrackingController + getJoint): " + err);
        }
    }
    var st = stableState[target.bucket];
    if (!st) {
        return;
    }

    if (label === st.lastRawLabel) {
        st.stableCount++;
    } else {
        st.lastRawLabel = label;
        st.stableCount = 0;
    }

    if (st.stableCount < script.minStableFrames) {
        return;
    }

    if (label === st.lastCommittedLabel) {
        return;
    }

    st.lastCommittedLabel = label;
    if (store) {
        try {
            store.putString(target.storageKey, label);
        } catch (e2) {
            print("[Gesture] storage put failed: " + e2);
        }
    }
    print("[Gesture] " + target.logPrefix + label + " (saved)");
}

function onUpdate() {
    try {
        var targets = getWatchTargets();
        var seen = {};

        for (var i = 0; i < targets.length; i++) {
            var rawLabel = "unknown";
            var heuristicLabel = "unknown";
            try {
                heuristicLabel = classifyForHand(targets[i].script);
                rawLabel = resolveLearnedClassification(targets[i].script, heuristicLabel);
            } catch (errC) {
                print("[Gesture] classify error (check HandTrackingController + getJoint): " + errC);
            }
            if (i === 0) {
                script.lastInstantLabel = rawLabel;
                if (typeof global !== "undefined") {
                    global.__aslGestureLabel = rawLabel;
                    global.__aslGestureLetter = gestureLabelToLetter(rawLabel);
                }
            }
            processTarget(targets[i], rawLabel);
            seen[targets[i].bucket] = true;
            if (script.debugHEachFrame && i === 0) {
                debugHForHand(targets[i].script, targets[i].bucket + ": ");
            }
            if (script.debugIEachFrame && i === 0) {
                debugIForHand(targets[i].script, targets[i].bucket + ": ");
            }
        }
        if (targets.length === 0) {
            script.lastInstantLabel = "";
            if (typeof global !== "undefined") {
                global.__aslGestureLabel = "";
                global.__aslGestureLetter = "";
            }
        }

        var resetIfMissing = function (bucket) {
            if (!seen[bucket] && stableState[bucket]) {
                stableState[bucket].stableCount = 0;
                stableState[bucket].lastRawLabel = null;
            }
        };

        if (script.handController) {
            resetIfMissing("override");
        } else if (script.watchHand === WATCH_BOTH) {
            resetIfMissing("left");
            resetIfMissing("right");
        } else if (script.watchHand === WATCH_LEFT) {
            resetIfMissing("left");
        } else if (script.watchHand === WATCH_RIGHT) {
            resetIfMissing("right");
        } else         if (script.watchHand === WATCH_ACTIVE) {
            resetIfMissing("active");
        }

        if (script.snapshotMode === 1 && targets.length > 0) {
            refreshHandSnapshotForScript(targets[0].script);
        }
    } catch (err) {
        print("[Gesture] onUpdate error: " + err);
    }
}

(function init() {
    if (store) {
        try {
            if (script.handController) {
                var prev = store.getString(STORAGE_KEY);
                if (prev && prev.length > 0) {
                    print("[Gesture] last saved type from storage: " + prev);
                }
            } else {
                if (script.watchHand === WATCH_LEFT || script.watchHand === WATCH_BOTH) {
                    var pl = store.getString(STORAGE_KEY_LEFT);
                    if (pl && pl.length > 0) {
                        print("[Gesture] last saved (left): " + pl);
                    }
                }
                if (script.watchHand === WATCH_RIGHT || script.watchHand === WATCH_BOTH) {
                    var pr = store.getString(STORAGE_KEY_RIGHT);
                    if (pr && pr.length > 0) {
                        print("[Gesture] last saved (right): " + pr);
                    }
                }
                if (script.watchHand === WATCH_ACTIVE) {
                    var pa = store.getString(STORAGE_KEY);
                    if (pa && pa.length > 0) {
                        print("[Gesture] last saved type from storage: " + pa);
                    }
                }
            }
        } catch (e0) {
            print("[Gesture] storage read failed: " + e0);
        }
    }
    script.createEvent("UpdateEvent").bind(onUpdate);

    script.createEvent("TapEvent").bind(function () {
        if (script.snapshotMode === 2) {
            var tt = getWatchTargets();
            if (tt.length < 1) {
                print("[Gesture] snapshot tap: no tracked hand");
                return;
            }
            var s = refreshHandSnapshotForScript(tt[0].script);
            print("[Gesture] snapshot tap: " + snapshotToJson(s));
            return;
        }
        if (script.snapshotMode === 3) {
            recordKnnSampleFromTap();
        }
        if (script.debugHOnTap) {
            var ht = getWatchTargets();
            if (ht.length < 1) {
                print("[Gesture][H debug] tap: no tracked hand");
            } else {
                debugHForHand(ht[0].script, ht[0].bucket + ": ");
            }
        }
        if (script.debugIOnTap) {
            var it = getWatchTargets();
            if (it.length < 1) {
                print("[Gesture][I debug] tap: no tracked hand");
            } else {
                debugIForHand(it[0].script, it[0].bucket + ": ");
            }
        }
    });

    script.printKnnDatasetSummary = function () {
        var d = loadKnnDataset();
        var by = {};
        var i;
        for (i = 0; i < d.samples.length; i++) {
            var lb = d.samples[i].label;
            by[lb] = (by[lb] || 0) + 1;
        }
        print("[Gesture] kNN samples=" + d.samples.length + " " + JSON.stringify(by));
    };
})();

