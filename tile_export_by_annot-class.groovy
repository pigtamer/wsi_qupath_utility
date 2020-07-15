/*
@Cunyuan JI
Run for project!
 */


import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathTileObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI

import java.awt.geom.AffineTransform

import static qupath.lib.gui.scripting.QPEx.*
import groovy.util.FileTreeBuilder

def server = getCurrentServer()
def downsample = 4.0
/*
Tree:

Project
    --DATA
        -- 01_17...
            -- IHC
                -- Annotations
                    -- txt file
                -- Tiles
                    -- Tumor
                        -- ROI_1...
                        -- ROI_2...
                    -- Blood
                    -- Other Tissues
            -- HE
                -- Tumor
                -- Blood
                -- Other Tissues
 */

def ProjBaseDir = "/Users/cunyuan/code/play/Kimura/"

/*
For WSIs:
    write annotation data to text file
    For annotations:
        For classes:
            Make directory in /Tiles
            For tiles:
                Write to directory
 */

def stain_types = ["HE", "IHC"]
// Process all HE images in the project.
def ImgName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

for (stain_name in stain_types) {
    if (ImgName.contains(stain_name)) {
        WSI_ID = ImgName
    } else {
        continue
    }

// Get the project & the requested image name

    def Objects = getAnnotationObjects()

//        new File(ProjBaseDir + "${stain_name}/${WSI_ID}").mkdirs()
    new File(ProjBaseDir + "${stain_name}/${WSI_ID}/Annotations/").mkdirs()
    def annotations = getAnnotationObjects().collect { new qupath.lib.objects.PathAnnotationObject(it.getROI(), it.getPathClass()) }
    print(annotations)
    new File(buildFilePath(ProjBaseDir + "${stain_name}/${WSI_ID}/Annotations/annotations-" + ImgName + '.txt')).withObjectOutputStream {
        it.writeObject(annotations)
    }
    // Write tiles
    for (pathObject in Objects) {
        print(pathObject)
        if (pathObject.hasChildren()) {
            className = pathObject.getProperties().get('displayedName')
            print(className)
            new File(ProjBaseDir + "${stain_name}/${WSI_ID}/Tiles/${className}").mkdirs()
            //export tiles in the dir
            tiles = pathObject.getChildObjects()
            def k = 0
            for (tile in tiles) {
                roi = tile.getROI()
                def sx = roi.getBoundsX()
                def sy = roi.getBoundsY()
                def sizey = roi.getBoundsHeight()
                def sizex = roi.getBoundsWidth()
                requestedTile = RegionRequest.createInstance(server.getPath(), downsample, roi)
                name = "${ProjBaseDir}${stain_name}/${WSI_ID}/Tiles/${className}/${k}_${WSI_ID}${stain_name}_(d=$downsample, x=$sx, y=$sy, w=$sizex, h=$sizey, z=${k}).tif".toString()
                print(roi)
                writeImageRegion(server, requestedTile, name)
                k += 1
            }
        }
    }
}

print 'Done!'


