from core.models import TranscriptionResult, DiarizationResult, SpeakerMapping

class MergeTranscript:
    TOLERANCE = 0.25  # seconds
    
    def merge(self, transcriptions: list[TranscriptionResult], diarizations: DiarizationResult, speaker_mapping: list[SpeakerMapping]) -> str:
        """
        Merge transcriptions with diarization by assigning each word to the best-matching speaker.
        Each word is assigned to exactly ONE speaker based on maximum overlap.
        Args:
            transcriptions (list[TranscriptionResult]): List of transcription segments with word timestamps.
            diarizations (DiarizationResult): Diarization result with speaker segments.
            speaker_mapping (SpeakerMapping)[Optional]: list of Mapping of diarization speaker labels to identified speaker names. If provided, use identified names instead of original labels.
        Returns:
            str: Formatted transcript with speaker labels.
        """
        # Flatten all words from all transcription segments
        all_words = []
        for transcription in transcriptions:
            for word in transcription.word_timestamps:
                all_words.append({
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"],
                    "midpoint": (word["start"] + word["end"]) / 2
                })
        
        # Assign each word to the best matching speaker
        word_speaker_assignments = []
        for word in all_words:
            best_speaker = self._assign_word_to_speaker(word, diarizations)
            if best_speaker:
                word_speaker_assignments.append({
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": best_speaker,
                    "text": word["text"]
                })
        
        # Sort by time and build transcript
        word_speaker_assignments.sort(key=lambda x: x["start"])
        if speaker_mapping is None:
            return self._build_transcript(word_speaker_assignments)
        else:
            return self._build_transcript(word_speaker_assignments, speaker_mapping)
    
    def _assign_word_to_speaker(self, word: dict, segments) -> str:
        """
        Assign a word to the speaker segment with the best overlap.
        Uses word midpoint for primary matching, falls back to overlap calculation.
        """
        word_midpoint = word["midpoint"]
        word_start = word["start"]
        word_end = word["end"]
        
        best_speaker = None
        best_overlap = 0
        
        for segment in segments:
            # Check if word midpoint falls within speaker segment (with tolerance)
            if segment.start_time - self.TOLERANCE <= word_midpoint <= segment.end_time + self.TOLERANCE:
                # Calculate overlap between word and speaker segment
                overlap_start = max(word_start, segment.start_time)
                overlap_end = min(word_end, segment.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = segment.speaker_label
        
        return best_speaker
    
    def _build_transcript(self, word_assignments: list[dict], speaker_mapping: list[SpeakerMapping]) -> str:
        """
        Build formatted transcript from word-speaker assignments.
        Groups consecutive words by the same speaker.
        """
        if not word_assignments:
            return ""
        
        transcript_lines = []
        current_speaker = None
        current_words = []
        
        for assignment in word_assignments:
            
            for s in speaker_mapping:
                if s.original_label == assignment["speaker"]:
                    speaker = s.identified_label
                    break
            word = assignment["text"]
            word_start = assignment["start"]
            word_end = assignment["end"]
            
            if speaker != current_speaker:
                # Save previous speaker's words
                if current_speaker and current_words:
                    transcript_lines.append(f"{current_speaker}: {' '.join(current_words)}")
                
                # Start new speaker section
                current_speaker = speaker
                current_words = [word]
            else:
                current_words.append(word)
        
        # Add final speaker's words
        if current_speaker and current_words:
            transcript_lines.append(f"{current_speaker}: {' '.join(current_words)}")
        
        return "\n".join(transcript_lines)
    