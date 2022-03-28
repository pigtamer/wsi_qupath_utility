/**
 * Script to export binary masks corresponding to all annotations of an image,
 * optionally along with extracted image regions.
 *
 * Note: Pay attention to the 'downsample' value to control the export resolution!
 *
 * @author Pete Bankhead
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage


import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathTileObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI
import ij.plugin.ImageCalculator
import qupath.imagej.tools.IJTools
import java.awt.geom.AffineTransform

import static qupath.lib.gui.scripting.QPEx.*
import groovy.util.FileTreeBuilder


// Get the main QuPath data structures
def imageData = getCurrentImageData()
// def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Request all objects from the hierarchy & filter only the annotations
// def annotations = hierarchy.getAnnotationObjects()

// Define downsample value for export resolution & output directory, creating directory if necessary
// def downsample = 4.0
def pathOutput = buildFilePath(QPEx.PROJECT_BASE_DIR, 'masks')
mkdirs(pathOutput)

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: masks will always be exported as PNG
def imageExportType = 'JPG'

// // Export each annotation
// annotations.each {
//     saveImageAndMask(pathOutput, server, it, downsample, imageExportType)
// }
print 'Done!'



/*
@Cunyuan JI
*/



// def server = getCurrentServer()
def downsample = 1.0000000

// 改成你的项目路径
def ProjBaseDir = '/Users/cunyuan/DATA/'


// ！！！
// ROI 边缘扩大的相对系尺寸。假设为0.5，那么左右多出宽为原图一半的边缘，图片尺寸扩大两倍
marginsize = 0.5
// 改成你的染色类型，和片子的文件名一致
def stain_types = ['HE', 'IHC']
// Process all HE images in the project.
def ImgName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

for (stain_name in stain_types) {
    if (ImgName.contains(stain_name)) {
        WSI_ID = ImgName
    } else {
        continue
    }
    def Objects = getAnnotationObjects()
    def annotations = getAnnotationObjects().collect { new qupath.lib.objects.PathAnnotationObject(it.getROI(), it.getPathClass()) }
    // Write tiles
    def o = 0
    for (pathObject in Objects) {
        print(pathObject)
        if (pathObject.hasChildren()) {
            className = pathObject.getProperties().get('displayedName')
            print(className)
            def k = 0
            if (className == 'Other') {
                //export tiles in the dir
                cells = pathObject.getChildObjects()
                // Export each annotation
                cells.each {
                    saveImageAndMask(pathOutput, server, it, downsample, imageExportType)
                }
            //     print 'Done!'
            //     for (cell in cells) {
            //         roi = cell.getROI()

            //         def sizey = roi.getBoundsHeight()
            //         def sizex = roi.getBoundsWidth()
            //         def sx = roi.getBoundsX()
            //         // - (int)(sizex * marginsize)
            //         def sy = roi.getBoundsY()

            //         requestedCell = RegionRequest.createInstance(server.getPath(), downsample, roi)

            //         name = "${ProjBaseDir}/${WSI_ID}_(d=$downsample, x=$sx, y=$sy, w=$sizex, h=$sizey, z=${k}).tif".toString()
            //         print(roi)
            //         writeImageRegion(server, requestedCell, name) //保存

            //         k += 1
            // }}
            }
        o += 1
        }
    }
}
// ROI变换函数
ROI transformROI(ROI roi, AffineTransform transform) {
    def shape = RoiTools.getShape(roi)
    // Should be able to use roi.getShape() - but there's currently a bug in it for rectangles/ellipses!
    shape2 = transform.createTransformedShape(shape)
    return RoiTools.getShapeROI(shape2, roi.getImagePlane(), 0.5)
}

/**
 * Save extracted image region & mask corresponding to an object ROI.
 *
 * @param pathOutput Directory in which to store the output
 * @param server ImageServer for the relevant image
 * @param pathObject The object to export
 * @param downsample Downsample value for the export of both image region & mask
 * @param imageExportType Type of image (original pixels, not mask!) to export ('JPG', 'PNG' or null)
 * @return
 */
def saveImageAndMask(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String imageExportType) {
    // Extract ROI & classification name
    def roi = pathObject.getNucleusROI()
    def pathClass = pathObject.getPathClass()
    def classificationName = pathClass == null ? 'None' : pathClass.toString()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region & mask'
        return
    }

    // Create a region from the ROI
    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

    // Create a name
    String name = String.format('%s_%s_(%.2f,%d,%d,%d,%d)',
            server.getMetadata().getName(),
            classificationName,
            region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create a mask using Java2D functionality
    // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
    def shape = RoiTools.getShape(roi)
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-region.getX(), -region.getY())
    g2d.fill(shape)
    g2d.dispose()


    ic = new ImageCalculator()
    
    def imp1 = IJTools.convertToUncalibratedImagePlus("Image",img)
    def imp2 = IJTools.convertToUncalibratedImagePlus("Mask", imgMask)
    def imgMulti = ic.run("Min create", imp1, imp2)
    def imgMultiExp = imgMulti.getBufferedImage()

//    // Create filename & export
//    if (imageExportType != null) {
//        def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
//        ImageIO.write(img, imageExportType, fileImage)
//    }
//    // Export the mask
//    def fileMask = new File(pathOutput, name + '-mask.png')
//    ImageIO.write(imgMask, 'PNG', fileMask)
    
    def fileMulti = new File(pathOutput, name + '-multi.png')
    ImageIO.write(imgMultiExp, 'PNG', fileMulti)



    // // Create filename & export
    // if (imageExportType != null) {
    //     def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
    //     ImageIO.write(img, imageExportType, fileImage)
    // }
    // // Export the mask
    // def fileMask = new File(pathOutput, name + '-mask.png')
    // ImageIO.write(imgMask, 'PNG', fileMask)



}