//
//  ViewController.swift
//  SANetTest
//
//  Created by 下川達也 on 2020/08/06.
//  Copyright © 2020 下川達也. All rights reserved.
//

import UIKit
import CoreML
let LIMITED_SIZE : CGFloat = 512

class ViewController: UIViewController {
    var imageView : UIImageView!
    var contentSize : CGSize!
    var vgg_model : sanet_vgg_512_16!
    var vgg_stylized_model : sanet_vgg_stylized_512_16!
    var transform_model : sanet_transform_512_16!
    var decoder_model : sanet_decoder_512_16!
    func resultToUIImage(_ resultArray:MLMultiArray){
        print(resultArray)
        guard let image = resultArray.image(min: 0, max: 1,axes: (1,2,3)) else{return}
        let resizeImage = image.resizeCrop(resize:contentSize)
        imageView = UIImageView(frame: view.frame)
        imageView.contentMode = .scaleAspectFit
        imageView.image = resizeImage
        view.addSubview(imageView)
    }
    public func preprocess(image: UIImage) -> MLMultiArray? {
        guard let pixels = image.pixelData()?.map({ (Double($0) / 1) }) else {
            return nil
        }
        guard let array = try? MLMultiArray(shape: [1,3,image.size.height, image.size.width] as [NSNumber], dataType: .float32) else {
            return nil
        }

        let r = pixels.enumerated().filter { $0.offset % 4 == 0 }.map { $0.element }
        let g = pixels.enumerated().filter { $0.offset % 4 == 1 }.map { $0.element }
        let b = pixels.enumerated().filter { $0.offset % 4 == 2 }.map { $0.element }
        print(r)
        let combination = r + g + b
        for (index, element) in combination.enumerated() {
            array[index] = NSNumber(value: element)
        }
        //imageView = UIImageView(frame: view.frame)
        //imageView.contentMode = .scaleAspectFit
        //imageView.image = array.image(min: 0, max: 255,axes: (1,2,3))
        //view.addSubview(imageView)

        return array
    }
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = .white
        contentSize = UIImage(named: "cat.jpg")!.size
        vgg_predict()
        // Do any additional setup after loading the view.
    }
    
    private func vgg_predict(){
        vgg_model = sanet_vgg_512_16()
        do{
            print(checkContentImageSize(UIImage(named: "cat.jpg")!).pixelBuffer())
            let inputs = sanet_vgg_512_16Input(content: checkContentImageSize(UIImage(named: "cat.jpg")!).pixelBuffer()!, style: checkStyleImageSize(UIImage(named: "goph.jpg")!).pixelBuffer()!)
            print("start")
            let start = Date()
            do{
                let options = MLPredictionOptions()
                options.usesCPUOnly = false
                let output = try vgg_model.prediction(input: inputs, options: options)//.predictions(inputs: [inputs])
                let content4 = output.input_32
                let content5 = output._163
                let style4 = output.input_76
                let style5 = output._300
                let end = Date().timeIntervalSince(start)
                print("vgg model prediction time => \(end)")
                vgg_model = nil
                vgg_stylized_predict(content4: content4 , style4: style4, content5: content5, style5: style5)
            }catch{
                vgg_model = nil
                print("error")
            }
        }catch{
            print("alpha設定時のエラー")
        }
    }
    
    private func vgg_stylized_predict(content4:MLMultiArray,style4:MLMultiArray,content5:MLMultiArray,style5:MLMultiArray){
        vgg_stylized_model = sanet_vgg_stylized_512_16()
        do{
            let inputs = sanet_vgg_stylized_512_16Input(content_4: content4 ,style_4: style4, content_5: content5, style_5: style5)
            let start = Date()
            do{
                let options = MLPredictionOptions()
                options.usesCPUOnly = false
                let output = try vgg_stylized_model.prediction(input: inputs, options: options)
                let new_content4 = output._20
                let new_content5 = output._29
                let new_style4 = output._38
                let new_style5 = output._47
                let end = Date().timeIntervalSince(start)
                print("vgg_stylized model prediction time => \(end)")
                vgg_stylized_model = nil
                transform_predict(content4: new_content4, style4: new_style4, content5: new_content5, style5: new_style5)
            }catch{
                vgg_stylized_model = nil
                print("error")
            }
        }
    }
    
    private func transform_predict(content4:MLMultiArray,style4:MLMultiArray,content5:MLMultiArray,style5:MLMultiArray){
        transform_model = sanet_transform_512_16()
        do{
            var alpha = try MLMultiArray(shape: [1], dataType: MLMultiArrayDataType.float32)
            alpha[0] = NSNumber(value: 1.0)
            let inputs = sanet_transform_512_16Input(content_4: content4, style_4: style4, content_5: content5, style_5: style5, alpha: alpha)
            let start = Date()
            do{
                let options = MLPredictionOptions()
                options.usesCPUOnly = false
                let output = try transform_model.prediction(input: inputs, options: options)
                let transform_output = output._627
                let end = Date().timeIntervalSince(start)
                print("transform model prediction time => \(end)")
                transform_model = nil
                decoder_predict(transform: transform_output)
            }
        }catch{
            transform_model = nil
            print("error")
        }
    }
   
    private func decoder_predict(transform:MLMultiArray){
        decoder_model = sanet_decoder_512_16()
        do{
            let inputs = sanet_decoder_512_16Input(sanet: transform)
            let start = Date()
            do{
                let options = MLPredictionOptions()
                options.usesCPUOnly = false
                let output = try decoder_model.prediction(input: inputs, options: options)
                let decoder_output = output._238
                let end = Date().timeIntervalSince(start)
                print("decoder model prediction time => \(end)")
                decoder_model = nil
                resultToUIImage(decoder_output)
            }catch{
                decoder_model = nil
                print("error")
            }
        }catch{
            print("error")
        }
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

    public func checkContentImageSize(_ image:UIImage)->UIImage{
        //return image.resize(to: CGSize(width: 256, height: 256))
        return image.resizeContent(to: CGSize(width: LIMITED_SIZE, height: LIMITED_SIZE))
    }
    public func checkStyleImageSize(_ image:UIImage)->UIImage{
        //return image.resize(to: CGSize(width: 256, height: 256))
        return image.resizeStyle(to: CGSize(width: LIMITED_SIZE, height: LIMITED_SIZE))
    }
}
