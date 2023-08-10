# Communication documentation

This file references the documentation needed to interact with the Echostra micro-service locally.

## Available TCP server requests

This section references all the requests a client can send to the main server.


### **Initiate an audio stream for transcription**

* **Description**  
The client requests the opening of a socket to which he can send audio data to be transcribed (speech-to-text operation).

* **Request**  
```json
{
    "command": "createSTTStream",
    "params": {
        "bit_depth": "integer", // Usually 8, 16, 24 or 32
        "sample_rate": "integer", // Usually 44100 or 48000
        "stereo": "boolean", // If false, condidered as mono
    }
}
```

* **Response**  
```json
{
    "statusCode": "integer", // 201: success ; 500: failure
    "message": "string",
    "port": "integer" // Port on which to connect to send the audio data
}
```

If successful, the sending client is considered a subscriber to the transcription results. In other words, it will receive the results on a regular basis, as soon as they become available via the connection to the TCP/IP server.

* **How to send audio data ?**  
To send audio data to the opened socket, it is very important that you transfer it raw, without any type of formatting.

* **Transcription results**  
```json
{
    "transcript": "string", // Transcription result
    "phrase_id": "integer", // ID of the transcribed phrase, the higher the more recent.
    "phrase_version": "integer" // Version of the transcribed phrase, the higher the more recent.
}
```