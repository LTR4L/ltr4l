/*
 * Copyright 2018 org.LTR4L
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ltr4l.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * NetworkShape parses the "layer" settings in the config file, and holds the information about
 * the number of nodes and their activation in a layer.
 * The default values are also provided.
 */
public class NetworkShape {

  @Deprecated
  public static NetworkShape parseSetting(String layers){
    if(layers == null){
      return new NetworkShape(new NetworkShape.LayerSetting(1, Activation.Type.Identity));
    }
    else{
      String[] layersInfo = layers.split(" ");
      NetworkShape nShape = new NetworkShape();
      for (int i = 0; i < layersInfo.length; i++) {
        String[] layerShape = layersInfo[i].split(",");
        Integer nodeNum = Integer.parseInt(layerShape[0]);
        //Default number of nodes is 1
        if (nodeNum == null || nodeNum < 0) {
          nodeNum = 1;
        }

        Activation actFunc = Activation.ActivationFactory.getActivator(layerShape[1]);
        nShape.add(nodeNum, actFunc);
      }
      return nShape;
    }
  }

  @Override
  public String toString(){
    return layerSettings.toString();
  }

  private final List<LayerSetting> layerSettings;

  public NetworkShape(LayerSetting... lss){
    this(Arrays.asList(lss));
  }

  public NetworkShape(List<LayerSetting> layerSettings){
    this.layerSettings = layerSettings;
  }

  public NetworkShape(){
    layerSettings = new ArrayList<>();
  }

  public void add(int num, Activation actFunc){
    layerSettings.add(new LayerSetting(num, actFunc));
  }

  public void add(LayerSetting layerSetting){
    layerSettings.add(layerSetting);
  }

  public int size(){
    return layerSettings.size();
  }

  public LayerSetting getLayerSetting(int index){
    return layerSettings.get(index);
  }

  public static class LayerSetting {

    private final int num;
    private final Activation actFunc;

    public LayerSetting(int num, Activation actFunc){
      this.num = num;
      this.actFunc = actFunc;
    }

    public int getNum(){
      return num;
    }

    public Activation getActivation(){
      return actFunc;
    }

    @Override
    public String toString(){
      String className = actFunc.getClass().getTypeName();
      if(className.contains("$")){
        int lastIdx = className.lastIndexOf("$");
        try{
          String subStr = className.substring(lastIdx + 1);
          int valueId = Integer.parseInt(subStr);
          String typeName =  Activation.Type.values()[valueId - 1].name();
          className = className.replaceAll("\\$",".").substring(0, lastIdx + 1) + typeName;
        }
        catch(NumberFormatException ignored){ }
      }
      return String.format("(%s,%d)", className, num);
    }
  }
}
