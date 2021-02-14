import React from 'react';
import './App.css';
import 'semantic-ui-css/semantic.min.css'
import {Button, Icon} from 'semantic-ui-react'
import * as tf from '@tensorflow/tfjs';
import {Tensor} from "@tensorflow/tfjs-core";

const flattenQueue = (queue: Float32Array[]): Float32Array => {
    const frameSize = queue[0].length;
    const freqData = new Float32Array(queue.length * frameSize);
    queue.forEach((data, i) => freqData.set(data, i * frameSize));
    return freqData;
}
const getInputTensorFromFrequencyData = (freqData: Float32Array, shape: number[]): tf.Tensor => {
    const vals = new Float32Array(tf.util.sizeFromShape(shape));
    // If the data is less than the output shape, the rest is padded with zeros.
    vals.set(freqData, vals.length - freqData.length);
    return tf.tensor(vals, shape);
}

let EPSILON: number | null = null;
const normalize = (x: tf.Tensor): tf.Tensor => {
    if (EPSILON == null) {
        EPSILON = tf.backend().epsilon();
    }
    return tf.tidy(() => {
        const {mean, variance} = tf.moments(x);
        // Add an EPSILON to the denominator to prevent division-by-zero.
        return tf.div(tf.sub(x, mean), tf.add(tf.sqrt(variance), EPSILON!));
    });
}

const FRAMES_IN_PATCH = 43
const FREQUENCY_BINS = 232

const startVad = async (): Promise<void> => {
    const model = await tf.loadLayersModel('model/model.json');
    model.summary();
    const audioContext = new AudioContext();

    const audioStream: MediaStream = await navigator.mediaDevices.getUserMedia({audio: true, video: false})
    const streamSource = audioContext.createMediaStreamSource(audioStream);

    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048
    analyser.smoothingTimeConstant = 0.0;
    streamSource.connect(analyser);
    const frame = new Float32Array(analyser.frequencyBinCount)
    const frames: Float32Array[] = []
    const onAudioFrame = async () => {
        analyser.getFloatFrequencyData(frame);
        if (frame[0] === -Infinity) {
            console.log("NEG INF!")
        } else {
            frames.push(frame.slice(0, FREQUENCY_BINS))
        }
        if (frames.length === FRAMES_IN_PATCH) {
            console.log("Patch collected")
            const freqData = flattenQueue(frames);
            const freqDataTensor = getInputTensorFromFrequencyData(
                freqData, [1, FRAMES_IN_PATCH, FREQUENCY_BINS, 1]);
            const normalizedX = normalize(freqDataTensor);
            const result = model.predict(normalizedX) as Tensor;
            const res: Float32Array = await result.data() as Float32Array;
            const prediction = res[0];
            const isSpeech = prediction > 0.32857817
            console.log({ isSpeech, confidence: res[0] });
            tf.dispose([freqDataTensor, normalizedX, result]);
            frames.length = 0
        }

    }

    setInterval(onAudioFrame.bind(this), 1000 * 1024 / 44100);
}


const App = () =>
    (
        <div className="App">
            <Button color='red' onClick={startVad}>
                <Icon name='microphone'/>
                Record
            </Button>
        </div>
    )

export default App;
