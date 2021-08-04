//Make sure you replace the default cell (or whatever) detection script with your own.


import qupath.lib.gui.QuPathGUI
//Use either "project" OR "outputFolder" to determine where your detection files will go
def project = QuPathGUI.getInstance().getProject().getBaseDirectory()
project = project.toString()+"\\detectionMeasurements\\"

//Make sure the output folder exists
// mkdirs(project)
//def outputFolder = "D:\\Results\\"
//mkdirs(outputFolder)


hierarchy = getCurrentHierarchy()
def annotations = getAnnotationObjects()
int i = 1
clearDetections()  //Just in case, so that the first detection list won't end up with extra stuff that was laying around

for (annotation in annotations)
{
hierarchy.getSelectionModel().clearSelection();
selectObjects{p -> p == annotation}
//**********************************************************************
//REPLACE THIS PART WITH YOUR OWN DETECTION CREATION SCRIPT

runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', '{"detectionImageBrightfield": "Optical Density Sum"}');

//************************************************************************

// saveDetectionMeasurements(project+" "+i+"detections.txt",)
i+=1
// clearDetections()
}

//Potentially replace all of the detections for viewing, after finishing the export

//selectAnnotations()
//runPlugin('qupath.imagej.detect.nuclei.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "maxBackground": 2.0,  "watershedPostProcess": true,  "excludeDAB": false,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');