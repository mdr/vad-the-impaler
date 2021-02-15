import React from 'react';
import './App.css';
import 'semantic-ui-css/semantic.min.css'
import {Button, Container, Grid, Header, Icon} from 'semantic-ui-react'
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
//
// const FRAMES_IN_PATCH = 43
// const FREQUENCY_BINS = 232
// const SPEECH_THRESHOLD = 0.3656232953071594
//

type AppState = {
    isRecording: boolean
    patchInfos: PatchInfo[]
}

interface PatchInfo {
    isSpeech: boolean
    confidence: number
}

class App extends React.Component<{}, AppState> {
    state: AppState = {
        isRecording: false,
        patchInfos: []
    }
    interval?: NodeJS.Timeout;

    render() {
        return (
            <div className="App">
                <p/>
                <Header as='h1'>VAD the Impaler</Header>
                {this.state.isRecording ?
                    <Button size='big' color='red' onClick={this.stopVad}>
                        <Icon name='stop'/>
                        Stop
                    </Button>
                    :
                    <Button size='big' color='red' onClick={this.startVad}>
                        <Icon name='microphone'/>
                        Start Detecting Speech
                    </Button>}
                <p/>
                {this.state.patchInfos.length > 0 &&
                <Container style={{border: "1px solid", padding: "5px"}}>
                    <Grid>
                        {
                            this.state.patchInfos.map((patchInfo, i) =>
                                <Grid.Column key={i}>
                                    {patchInfo.isSpeech ? <Icon size='big' name='chat' color='teal'/> :
                                        <Icon size='big' name='window minimize'/>}
                                    <p>
                                        {Math.round(100 * patchInfo.confidence)}%
                                    </p>
                                </Grid.Column>
                            )
                        }
                    </Grid>
                </Container>
                }
            </div>
        );
    }
    stopVad = () => {
        this.setState({
            isRecording: false,
        });
        if (this.interval)
            clearInterval(this.interval)
    }

    startVad = async (): Promise<void> => {
        this.setState({
            isRecording: true,
        });
        const response = await fetch("model/metadata.json")
        const { frequencyBins, frames: framesInPatch, threshold: speechThreshold} = await response.json()
        const model = await tf.loadLayersModel('model/model.json');
        model.summary();
        // await tf.setBackend('cpu')
        console.log(tf.getBackend());
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
            if (frame[0] !== -Infinity) {
                frames.push(frame.slice(0, frequencyBins))
            }
            if (frames.length === framesInPatch) {
                console.time("patch classified")
                const freqData = flattenQueue(frames);
                const freqDataTensor = getInputTensorFromFrequencyData(
                    freqData, [1, framesInPatch, frequencyBins, 1]);
                const normalizedX = normalize(freqDataTensor);
                const result = model.predict(normalizedX) as Tensor;
                const res: Float32Array = await result.data() as Float32Array;
                const prediction = res[0];
                const isSpeech = prediction > speechThreshold
                const patchInfo = {isSpeech, confidence: res[0]}
                this.setState(state => ({
                    patchInfos: [...state.patchInfos, patchInfo]
                }))
                console.log(patchInfo);
                tf.dispose([freqDataTensor, normalizedX, result]);
                frames.length = 0
                console.timeEnd("patch classified")
            }

        }

        const frameDuration = 1000.0 * 1024 / 44100;
        this.interval = setInterval(onAudioFrame.bind(this), frameDuration);
    }

}

export default App;
