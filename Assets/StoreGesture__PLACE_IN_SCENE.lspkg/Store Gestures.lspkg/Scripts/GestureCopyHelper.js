// GestureCopyHelper.js
// Version: 0.0.1
// Event: Lens Initialized
// Description: Copy gesture to another object while save to persistent storage, best to use with 3D body and hand tracking, but can be used with any 3d objects

//@ui{"label":"Try this with any 3D objects, like 3D Body Tracking!"}
//@ui{"widget":"separator"}
//@ui{"label":"Make sure CopyFromObj and CopyToObj"}
//@ui{"label":"has the same hierarchy and child names."}
//@ui{"widget":"separator"}
//@input SceneObject copyFromObj
//@input SceneObject copyToObj
//@input int copyMethod = 0 {"widget":"combobox","values":[{"value":"0","label":"OnUpdate"},{"value":"1","label":"OnTap"}]}

var store = global.persistentStorageSystem.store;
var jointPairs = {};

function JointPair(fromObj, toObj){
    this.copyFromTransform = fromObj.getTransform();
    this.copyToTransform = toObj.getTransform();
    this.localRotation = this.copyFromTransform.getLocalRotation();
    //this.position = this.copyFromTransform.getWorldPosition();
}

JointPair.prototype.update = function(){
    this.localRotation = this.copyFromTransform.getLocalRotation();
    //this.position = this.copyFromTransform.getWorldPosition();

    if(script.copyMethod == 0){
        this.copyToTransform.setLocalRotation(this.localRotation);
    }

}

function getJointPairRecursive(parentA, parentB){
    for(var i = 0; i < parentA.getChildrenCount(); i ++){
        if(parentB.getChild(i).name == parentA.getChild(i).name){
            jointPairs[parentA.getChild(i).name] = new JointPair(parentA.getChild(i), parentB.getChild(i));
            if(parentA.getChild(i).getChildrenCount() > 0){
                getJointPairRecursive(parentA.getChild(i), parentB.getChild(i))
            }
        }
    }
}

function initialize(){
    getJointPairRecursive(script.copyFromObj, script.copyToObj);
    
    //Uncomment this line to let project remember last saved gesture!
    script.applyGesture();
    
    script.createEvent("UpdateEvent").bind(onUpdate);
}

function onUpdate(){
    for(var i in jointPairs){
        jointPairs[i].update();
    }
    
    //Uncomment the following to save gesture data at any time to persistent storage
//    if(script.copyMethod == 1){
//        script.recordGesture();
//    }
}


script.createEvent("TapEvent").bind(function(){
    if(script.copyMethod == 1){
        //Here we are using persistent storage to copy and paste gestures
        //Feel free to call these 2 functions in any other scripts, or use applyGesture() on start
        //to remember gestures left in the last save.
        script.recordGesture();
        script.applyGesture();
    }
});

script.recordGesture = function(){
    var rotationString= "";
    var positionString = "";
    
    for(var i in jointPairs){
        var rot = jointPairs[i].localRotation;
        var rotLabel = i;
        if(script.copyData != 1){
            rotationString += rotLabel;
            rotationString += rot.toString();
            store.putString(rotLabel, rot.toString());
        }

    }
}

script.applyGesture = function(){
    for(var i in jointPairs){
        var rotString = store.getString(i);
        
        var rotx = parseFloat(rotString.substring(rotString.indexOf("x") + 3, rotString.indexOf("y") - 2));
        var roty = parseFloat(rotString.substring(rotString.indexOf("y") + 3, rotString.indexOf("z") - 2));
        var rotz = parseFloat(rotString.substring(rotString.indexOf("z") + 3, rotString.indexOf("w") - 2));
        var rotw = parseFloat(rotString.substring(rotString.indexOf("w") + 3, rotString.indexOf("}")));
        var newRot = new quat(rotw, rotx, roty, rotz);
        jointPairs[i].copyToTransform.setLocalRotation(newRot);

    }
   
}

function getJointObject(jointName){
    var attachedObj = script.handTracking.getAttachedObjects(jointName);
    if (attachedObj.length < 1) {
        print("ERROR! No Joints with name of " + jointName);
        return;
    }
    return attachedObj[0];
}

function getJointRotation(jointObject){
    return jointObject.getTransform().getLocalRotation();
}


function findChildObjectRecursive(childName, parentObj){
    var childObj = null;
    for(var i = 0; i < parentObj.getChildrenCount(); i ++){
        if(jointName == parentObj.getChild(i).name){
            childObj = parentObj.getChild(i);
            break;
        }else{
            childObj = findChildObjectRecursive(jointName, parentObj.getChild(i));
            if(childObj){
                break;
            }
        }
    }
    return childObj;
}

initialize();