import React, { useState, useEffect, useRef } from 'react';
import API_URL from '../config';
import {
  Avatar,
  Box,
  IconButton,
  Stack,
  TextField,
  Typography,
  Chip,
  Alert,
  Snackbar,
  Select,
  MenuItem,
  FormControl,
} from '@mui/material';
import {
  SmartToy as BotIcon,
  Send as SendIcon,
  AttachFile as AttachFileIcon,
  Mic as MicIcon,
  Stop as StopIcon,
  VolumeUp as VolumeUpIcon,
  VolumeOff as VolumeOffIcon,
  Translate as TranslateIcon,
} from '@mui/icons-material';
import ChartComponent from './ChartComponent';

interface ChartData {
  type: 'bar' | 'line' | 'pie' | 'doughnut';
  title: string;
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
    fill?: boolean;
  }>;
  chart_url?: string;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isTyping?: boolean;
  citations?: string[];
  isBookmarked?: boolean;
  confidence?: number;
  sources?: any[];
  chartData?: ChartData;
  hasAudio?: boolean;
  audioPlayed?: boolean;
}

interface Document {
  document_id: string;
  document_name: string;
  upload_time: string;
  pages: number;
  word_count: number;
  document_type: string;
  status?: 'processing' | 'ready' | 'error';
  summary?: string;
}

interface ChatInterfaceProps {
  selectedDocument: Document | null;
  messages: Message[];
  inputMessage: string;
  isTyping: boolean;
  isRecording: boolean;
  isPlaying: string | null;
  darkMode: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setInputMessage: React.Dispatch<React.SetStateAction<string>>;
  handleSendMessage: () => void;
  toggleRecording: () => void;
  toggleAudioPlayback: (messageId: string) => void;
  bookmarkMessage: (messageId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  inputMessage,
  isTyping,
  messagesEndRef,
  setMessages,
  setInputMessage,
  handleSendMessage,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState('');
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null);
  const [audioError, setAudioError] = useState('');
  const recognitionRef = useRef<any>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
          }
        }

        if (finalTranscript) {
          setInputMessage(prev => prev + finalTranscript);
        }
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setRecordingError(`Error: ${event.error}`);
        setIsRecording(false);
      };

      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [setInputMessage]);

  // Auto-play audio for messages with hasAudio flag
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && 
        lastMessage.sender === 'bot' && 
        lastMessage.hasAudio && 
        !lastMessage.audioPlayed &&
        !lastMessage.isTyping) {
      // Auto-play audio after a short delay
      const timer = setTimeout(() => {
        playMessageAudio(lastMessage.id, lastMessage.text);
        // Mark as played (you'd need to update the message in parent component)
      }, 500);
      
      return () => clearTimeout(timer);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages]);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      setRecordingError('Speech recognition is not supported in your browser');
      return;
    }

    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
        setRecordingError('');
      } catch (error) {
        console.error('Failed to start recording:', error);
        setRecordingError('Failed to start recording');
      }
    }
  };

  const playMessageAudio = async (messageId: string, text: string) => {
    try {
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      if (playingMessageId === messageId) {
        // If clicking the same message, stop playback
        setPlayingMessageId(null);
        return;
      }

      setPlayingMessageId(messageId);
      setAudioError('');

      // Call TTS API
      const response = await fetch(`${API_URL}/text-to-speech`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          lang: 'en',
          slow: false
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate speech');
      }

      // Create audio from blob
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        setPlayingMessageId(null);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = () => {
        setPlayingMessageId(null);
        setAudioError('Failed to play audio');
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();

    } catch (error) {
      console.error('Error playing audio:', error);
      setAudioError('Failed to generate or play audio');
      setPlayingMessageId(null);
    }
  };

  const translateMessage = async (messageId: string, text: string, targetLang: string) => {
    if (targetLang === 'en') return; // Assuming English is the default and no translation needed

    try {
      const response = await fetch(`${API_URL}/translate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          target_lang: targetLang
        })
      });

      if (!response.ok) {
        throw new Error('Translation failed');
      }

      const data = await response.json();
      
      // Update message text in the parent component
      setMessages(prev => prev.map(msg => 
        msg.id === messageId ? { ...msg, text: data.translated_text } : msg
      ));

    } catch (error) {
      console.error('Error translating message:', error);
      setAudioError('Failed to translate message');
    }
  };

  return (
    <Box sx={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      bgcolor: 'background.default',
      position: 'relative'
    }}>
      {/* Messages Area */}
      <Box sx={{
        flexGrow: 1,
        overflow: 'auto',
        py: 4,
        bgcolor: 'background.default'
      }}>
        <Box sx={{ 
          maxWidth: { xs: '100%', sm: '100%', md: '800px', lg: '900px' }, 
          mx: 'auto',
          px: { xs: 2, sm: 3, md: 4 }
        }}>
          <Stack spacing={3}>
            {messages.map((message) => (
              <Box
                key={message.id}
                sx={{
                  display: 'flex',
                  gap: 2,
                  alignItems: 'flex-start',
                  width: '100%'
                }}
              >
                {/* Avatar */}
                <Avatar
                  sx={{
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    width: 32,
                    height: 32,
                    flexShrink: 0,
                    fontSize: '0.875rem',
                    fontWeight: 600
                  }}
                >
                  {message.sender === 'bot' ? (
                    <BotIcon sx={{ fontSize: 18, color: 'primary.contrastText' }} />
                  ) : (
                    'U'
                  )}
                </Avatar>

                {/* Message Content */}
                <Box sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 0.5,
                }}>
                  <Box
                    sx={{
                      py: 1,
                      color: 'text.primary',
                    }}
                  >
                    {/* Audio indicator badge */}
                    {message.hasAudio && message.sender === 'bot' && (
                      <Chip
                        icon={<VolumeUpIcon sx={{ fontSize: 14 }} />}
                        label="Audio Response"
                        size="small"
                        sx={{
                          mb: 1,
                          height: 24,
                          fontSize: '0.75rem',
                          bgcolor: 'primary.main',
                          color: 'primary.contrastText',
                          '& .MuiChip-icon': {
                            color: 'primary.contrastText'
                          }
                        }}
                      />
                    )}

                    <Typography
                      variant="body1"
                      sx={{
                        whiteSpace: 'pre-wrap',
                        lineHeight: 1.7,
                        fontSize: '1rem',
                        fontWeight: 400,
                        color: 'text.primary'
                      }}
                    >
                      {message.text}
                    </Typography>

                    {message.chartData && (
                      <Box sx={{ mt: 2 }}>
                        <ChartComponent chartData={message.chartData} />
                      </Box>
                    )}

                    {message.citations && message.citations.length > 0 && (
                      <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                        <Typography variant="caption" sx={{ opacity: 0.7, display: 'block', mb: 1, fontWeight: 600 }}>
                          Sources:
                        </Typography>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap">
                          {message.citations.map((citation, index) => (
                            <Chip
                              key={index}
                              label={`${index + 1}`}
                              size="small"
                              variant="outlined"
                              sx={{
                                height: 22,
                                fontSize: '0.7rem',
                                borderColor: 'divider',
                                '&:hover': {
                                  bgcolor: 'action.hover'
                                }
                              }}
                            />
                          ))}
                        </Stack>
                      </Box>
                    )}
                  </Box>

                  <Stack
                    direction="row"
                    alignItems="center"
                    spacing={1}
                    sx={{
                      mt: 1
                    }}
                  >
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{
                        fontSize: '0.75rem',
                      }}
                    >
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </Typography>

                    {/* Voice playback button for bot messages */}
                    {message.sender === 'bot' && !message.isTyping && (
                      <IconButton
                        size="small"
                        onClick={() => playMessageAudio(message.id, message.text)}
                        sx={{
                          width: 28,
                          height: 28,
                          color: playingMessageId === message.id ? 'primary.main' : 'text.secondary',
                          '&:hover': {
                            bgcolor: 'action.hover',
                          }
                        }}
                      >
                        {playingMessageId === message.id ? (
                          <VolumeOffIcon sx={{ fontSize: 16 }} />
                        ) : (
                          <VolumeUpIcon sx={{ fontSize: 16 }} />
                        )}
                      </IconButton>
                    )}

                    {/* Translation Dropdown for bot messages */}
                    {message.sender === 'bot' && !message.isTyping && (
                      <FormControl size="small" sx={{ minWidth: 100 }}>
                        <Select
                          value=""
                          displayEmpty
                          onChange={(e) => translateMessage(message.id, message.text, e.target.value as string)}
                          sx={{
                            height: 28,
                            fontSize: '0.75rem',
                            '& .MuiSelect-select': {
                              py: 0.5,
                              px: 1,
                              display: 'flex',
                              alignItems: 'center',
                              gap: 0.5
                            },
                            bgcolor: 'action.hover',
                            border: 'none',
                            '& fieldset': { border: 'none' }
                          }}
                          renderValue={(selected) => (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <TranslateIcon sx={{ fontSize: 14 }} />
                              Translate
                            </Box>
                          )}
                        >
                          <MenuItem value="en" disabled>English</MenuItem>
                          <MenuItem value="Spanish">Spanish</MenuItem>
                          <MenuItem value="French">French</MenuItem>
                          <MenuItem value="German">German</MenuItem>
                          <MenuItem value="Hindi">Hindi</MenuItem>
                          <MenuItem value="Telugu">Telugu</MenuItem>
                          <MenuItem value="Chinese">Chinese</MenuItem>
                          <MenuItem value="Japanese">Japanese</MenuItem>
                        </Select>
                      </FormControl>
                    )}
                  </Stack>
                </Box>
              </Box>
            ))}

            {/* Typing Indicator */}
            {isTyping && (
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', width: '100%' }}>
                <Avatar
                  sx={{
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    width: 32,
                    height: 32,
                    flexShrink: 0,
                    fontSize: '0.875rem',
                    fontWeight: 600
                  }}
                >
                  <BotIcon sx={{ fontSize: 18, color: 'primary.contrastText' }} />
                </Avatar>
                <Box sx={{ flex: 1, py: 1 }}>
                  <Stack direction="row" spacing={0.5} alignItems="center">
                    {[0, 1, 2].map((i) => (
                      <Box
                        key={i}
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: 'text.secondary',
                          animation: 'bounce 1.4s ease-in-out infinite',
                          animationDelay: `${i * 0.2}s`,
                          '@keyframes bounce': {
                            '0%, 60%, 100%': {
                              transform: 'translateY(0)',
                              opacity: 0.3,
                            },
                            '30%': {
                              transform: 'translateY(-6px)',
                              opacity: 1,
                            }
                          }
                        }}
                      />
                    ))}
                  </Stack>
                </Box>
              </Box>
            )}

            <div ref={messagesEndRef} />
          </Stack>
        </Box>
      </Box>

      {/* Input Area */}
      <Box
        sx={{
          py: 3,
          px: { xs: 2, sm: 3, md: 4 },
          borderTop: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
        }}
      >
        <Box sx={{ 
          maxWidth: { xs: '100%', sm: '100%', md: '800px', lg: '900px' }, 
          mx: 'auto'
        }}>
          <Box sx={{
            display: 'flex',
            gap: 1.5,
            alignItems: 'flex-end',
          }}>
            <IconButton
              size="medium"
              sx={{
                color: 'text.secondary',
                width: 38,
                height: 38,
                '&:hover': {
                  color: 'primary.main',
                  bgcolor: 'action.hover'
                }
              }}
            >
              <AttachFileIcon sx={{ fontSize: 20 }} />
            </IconButton>

            <TextField
              fullWidth
              multiline
              maxRows={6}
              variant="outlined"
              placeholder="Message IntelliDoc..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '12px',
                  bgcolor: 'background.default',
                  fontSize: '1rem',
                  py: 1.5,
                  px: 2,
                  '& fieldset': {
                    borderColor: 'divider',
                  },
                  '&:hover fieldset': {
                    borderColor: 'text.secondary',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: 'text.primary',
                    borderWidth: 2,
                  }
                }
              }}
            />

            <IconButton
              size="medium"
              onClick={toggleRecording}
              sx={{
                color: isRecording ? '#f44336' : 'text.secondary',
                bgcolor: isRecording ? 'rgba(244, 67, 54, 0.1)' : 'transparent',
                width: 38,
                height: 38,
                animation: isRecording ? 'pulse 1.5s ease-in-out infinite' : 'none',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.6 }
                },
                '&:hover': {
                  color: isRecording ? '#d32f2f' : 'primary.main',
                  bgcolor: isRecording ? 'rgba(244, 67, 54, 0.2)' : 'action.hover',
                }
              }}
            >
              {isRecording ? <StopIcon sx={{ fontSize: 20 }} /> : <MicIcon sx={{ fontSize: 20 }} />}
            </IconButton>

            <IconButton
              onClick={handleSendMessage}
              disabled={!inputMessage.trim()}
              sx={{
                bgcolor: inputMessage.trim() ? 'text.primary' : 'action.disabledBackground',
                color: 'white',
                width: 40,
                height: 40,
                '&:hover': {
                  bgcolor: inputMessage.trim() ? 'text.secondary' : 'action.disabledBackground',
                },
                '&.Mui-disabled': {
                  bgcolor: 'action.disabledBackground',
                  color: 'action.disabled',
                },
                transition: 'all 0.2s ease-in-out',
              }}
            >
              <SendIcon sx={{ fontSize: 20 }} />
            </IconButton>
          </Box>
        </Box>

        {/* Recording Indicator */}
        {isRecording && (
          <Box sx={{
            mt: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 1,
            color: '#f44336'
          }}>
            <Box sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: '#f44336',
              animation: 'blink 1s ease-in-out infinite',
              '@keyframes blink': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.2 }
              }
            }} />
            <Typography variant="caption" color="error" sx={{ fontSize: '0.8rem', fontWeight: 500 }}>
              Recording... Speak now
            </Typography>
          </Box>
        )}
      </Box>

      {/* Error Snackbars */}
      <Snackbar
        open={!!recordingError}
        autoHideDuration={4000}
        onClose={() => setRecordingError('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setRecordingError('')}
          severity="error"
          sx={{ width: '100%' }}
        >
          {recordingError}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!audioError}
        autoHideDuration={4000}
        onClose={() => setAudioError('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setAudioError('')}
          severity="error"
          sx={{ width: '100%' }}
        >
          {audioError}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ChatInterface;
