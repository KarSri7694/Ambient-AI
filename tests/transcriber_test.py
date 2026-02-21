from faster_whisper import BatchedInferencePipeline, WhisperModel
import time

HIN2HINGLISH = "Hin2Hinglish-ct2/"
def infer(filepath):
    model = WhisperModel(HIN2HINGLISH, device="cuda", compute_type="int8_float16")
    batched_model = BatchedInferencePipeline(model)
    start_time = time.time()
    segments,_ = batched_model.transcribe(filepath, word_timestamps=True, chunk_length=30, batch_size=8, vad_filter=True)

    for segment in segments:
        print(f"timestamps: [{segment.start}, {segment.end}] text: {segment.text}")
    end_time = time.time() - start_time
    print(f"Transcription completed in {end_time:.4f} seconds")    
infer("uploads/From_Logic_Gates_to_LLMs__Unpacking_the_Evolution_of_AI_and_Its_Mind-Bending_Foundations.m4a")