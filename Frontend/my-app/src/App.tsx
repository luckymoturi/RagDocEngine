import API_URL from './config';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';

import {
  Avatar,
  Box,
  Button,
  Chip,
  CssBaseline,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Drawer,
  FormControl,
  FormControlLabel,
  IconButton,
  MenuItem,
  Paper,
  Select,
  Snackbar,
  Stack,
  Switch,
  TextField,
  ThemeProvider,
  Toolbar,
  Typography,
  createTheme,
  useMediaQuery,
  Alert,
  LinearProgress
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Search as SearchIcon,
  Description as DescriptionIcon,
  Analytics as AnalyticsIcon,
  Menu as MenuIcon,
  Close as CloseIcon,
  Chat as ChatIcon,
  Settings as SettingsIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon
} from '@mui/icons-material';
import { ChatInterface, DocumentsView, AnalyticsView } from './components';

const createAppTheme = (darkMode: boolean) => createTheme({
  palette: {
    mode: darkMode ? 'dark' : 'light',
    primary: {
      main: '#BFC9D1',
      contrastText: '#0F172A',
    },
    background: {
      default: darkMode ? '#0F172A' : '#F8FAFC', // Slate-50 like background
      paper: darkMode ? '#1E293B' : '#FFFFFF',
    },
    text: {
      primary: darkMode ? '#F8FAFC' : '#0F172A', // Slate-900 like text
      secondary: darkMode ? '#94A3B8' : '#334155', // Darker Slate-700 (was 64748B)
    },
    divider: darkMode ? 'rgba(255,255,255,0.1)' : '#E2E8F0',
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          textTransform: 'none',
          fontWeight: 500,
          boxShadow: 'none',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: 'none',
            backgroundColor: darkMode ? 'rgba(255,255,255,0.05)' : '#F1F5F9',
            color: '#0F172A',
          },
        },
        containedPrimary: {
          backgroundColor: '#BFC9D1',
          color: '#0F172A',
          '&:hover': {
            backgroundColor: '#A8B4BD',
          },
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          margin: '2px 8px',
          '&.Mui-selected': {
            backgroundColor: darkMode ? '#334155' : '#F1F5F9',
            color: '#0F172A',
            '&:hover': {
              backgroundColor: darkMode ? '#475569' : '#E2E8F0',
            },
          },
          '&:hover': {
            backgroundColor: darkMode ? '#1E293B' : '#F8FAFC',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: darkMode
            ? '0 4px 6px -1px rgba(0, 0, 0, 0.2)'
            : '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
          border: '1px solid',
          borderColor: darkMode ? 'rgba(255,255,255,0.1)' : '#E2E8F0',
          transition: 'all 0.2s ease-in-out',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundImage: 'none',
          border: '1px solid',
          borderColor: darkMode ? 'rgba(255,255,255,0.1)' : '#E2E8F0',
        },
      },
    },
  },
});

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

interface Analytics {
  basic_info: any;
  content_analysis: any;
  readability: any;
  key_statistics: any;
  topics: string[];
  sentiment: any;
}

const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);
  const theme = createAppTheme(darkMode);
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // Core state
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [chatWithAllDocs, setChatWithAllDocs] = useState(false); // Multi-document chat mode

  // UI state
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [currentView, setCurrentView] = useState<'chat' | 'documents' | 'analytics'>('documents');
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'info' | 'warning'
  });

  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AI document assistant. Upload a document and start asking questions about it. I can help you with summaries, specific queries, detailed answers with citations, and even create visualizations!\n\n**Try asking me to:**\n• Summarize your document\n• Answer specific questions\n• Create charts and graphs from your data\n• Analyze document statistics\n• Visualize key information',
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const isRecording = false;
  const isPlaying: string | null = null;

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [keywords, setKeywords] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);

  // Dialog state
  const [uploadDialog, setUploadDialog] = useState(false);
  const [settingsDialog, setSettingsDialog] = useState(false);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto close drawer on mobile
  useEffect(() => {
    if (isMobile) {
      setDrawerOpen(false);
    } else {
      setDrawerOpen(true);
    }
  }, [isMobile]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const showSnackbar = useCallback((message: string, severity: 'success' | 'error' | 'info' | 'warning') => {
    setSnackbar({ open: true, message, severity });
  }, []);

  // Document management
  const handleDocumentUpload = useCallback(async (files: File[]) => {
    setLoading(true);

    try {
      for (const file of files) {
        // Add document with processing status
        const newDoc: Document = {
          document_id: Date.now().toString(),
          document_name: file.name,
          upload_time: new Date().toISOString(),
          pages: 0,
          word_count: 0,
          document_type: file.name.split('.').pop()?.toUpperCase() || 'UNKNOWN',
          status: 'processing',
          summary: 'Processing document...'
        };

        setDocuments(prev => [...prev, newDoc]);

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/upload-pdf`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();

        // Update document with success status
        setDocuments(prev => prev.map(doc =>
          doc.document_id === newDoc.document_id
            ? {
              ...doc,
              document_id: data.document_id,
              pages: data.pages || 0,
              status: 'ready' as const,
              summary: data.summary || 'Document processed successfully'
            }
            : doc
        ));

        // Set as selected document if it's the first one
        if (!selectedDocument) {
          setSelectedDocument({
            document_id: data.document_id,
            document_name: file.name,
            upload_time: new Date().toISOString(),
            pages: data.pages || 0,
            word_count: 0,
            document_type: file.name.split('.').pop()?.toUpperCase() || 'UNKNOWN',
            status: 'ready',
            summary: data.summary
          });
        }
      }

      setUploadDialog(false);
      showSnackbar(`${files.length} document(s) uploaded successfully`, 'success');

    } catch (error) {
      console.error('Error uploading document:', error);
      showSnackbar('Error uploading document', 'error');

      // Update failed documents
      setDocuments(prev => prev.map(doc =>
        doc.status === 'processing'
          ? { ...doc, status: 'error' as const, summary: 'Upload failed' }
          : doc
      ));
    } finally {
      setLoading(false);
    }
  }, [selectedDocument, showSnackbar]);

  // Chat functionality
  const handleSendMessage = useCallback(async () => {
    // Allow chat if either a document is selected OR chatWithAllDocs is enabled
    if (!inputMessage.trim() || (!selectedDocument && !chatWithAllDocs)) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputMessage,
          document_id: chatWithAllDocs ? null : selectedDocument?.document_id,  // null = search all docs
          compare_mode: false  // Disabled - show only best answer
        })
      });

      if (!response.ok) {
        throw new Error(`Chat failed: ${response.statusText}`);
      }

      const data = await response.json();

      // Use the best answer (RAG by default)
      let messageText = data.response;

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: messageText,
        sender: 'bot',
        timestamp: new Date(),
        citations: data.sources?.map((source: any) => source.chunk_text?.substring(0, 100) + '...') || [],
        confidence: data.confidence,
        sources: data.sources,
        chartData: data.chart_data,
        hasAudio: data.has_audio || false,
        audioPlayed: false
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error processing your message. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      showSnackbar('Error sending message', 'error');
    } finally {
      setIsTyping(false);
    }
  }, [inputMessage, selectedDocument, chatWithAllDocs, showSnackbar]);


  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      handleDocumentUpload(acceptedFiles);
    }
  }, [handleDocumentUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: true
  });

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/documents`);
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.statusText}`);
      }
      const data = await response.json();
      const fetchedDocs = (data.documents || []).map((doc: any) => ({
        ...doc,
        status: 'ready' as const
      }));

      setDocuments(fetchedDocs);

      if (fetchedDocs.length > 0 && !selectedDocument) {
        setSelectedDocument(fetchedDocs[0]);
      }

    } catch (error) {
      console.error('Error fetching documents:', error);
      showSnackbar('Error loading documents', 'error');
    } finally {
      setLoading(false);
    }
  }, [selectedDocument, showSnackbar]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const fetchAnalytics = useCallback(async (documentId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/document/${documentId}/analytics`);
      if (!response.ok) {
        throw new Error(`Analytics failed: ${response.statusText}`);
      }
      const data = await response.json();
      setAnalytics(data);
      setCurrentView('analytics');
      showSnackbar('Analytics loaded successfully', 'success');
    } catch (error) {
      console.error('Error fetching analytics:', error);
      showSnackbar('Error loading analytics', 'error');
    } finally {
      setLoading(false);
    }
  }, [showSnackbar]);

  const deleteDocument = useCallback(async (documentId: string) => {
    if (!window.confirm('Are you sure you want to delete this document?')) {
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/document/${documentId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(`Delete failed: ${response.statusText}`);
      }

      setDocuments(prev => prev.filter(doc => doc.document_id !== documentId));

      if (selectedDocument?.document_id === documentId) {
        const remainingDocs = documents.filter(doc => doc.document_id !== documentId);
        setSelectedDocument(remainingDocs.length > 0 ? remainingDocs[0] : null);
        if (remainingDocs.length === 0) {
          setCurrentView('documents');
        }
      }

      showSnackbar('Document deleted successfully', 'success');
    } catch (error) {
      console.error('Error deleting document:', error);
      showSnackbar('Error deleting document', 'error');
    } finally {
      setLoading(false);
    }
  }, [selectedDocument, documents, showSnackbar]);

  const handleSearch = useCallback(async () => {
    if (!keywords.trim() || !selectedDocument) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/keyword-search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keywords: keywords.split(',').map(k => k.trim()),
          document_id: selectedDocument.document_id
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      setSearchResults(data.results || []);
      showSnackbar('Search completed', 'success');
    } catch (error) {
      console.error('Search error:', error);
      showSnackbar('Error performing search', 'error');
    } finally {
      setLoading(false);
    }
  }, [keywords, selectedDocument, showSnackbar]);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      handleDocumentUpload(Array.from(files));
    }
  }, [handleDocumentUpload]);

  // Sidebar component
  const SidebarContent = () => (
    <Box sx={{
      width: '100%',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      bgcolor: 'background.paper',
    }}>
      {/* Header */}
      <Box sx={{
        p: 2,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: darkMode ? 'transparent' : '#FFFFFF',
      }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <Box sx={{
              width: 40,
              height: 40,
              borderRadius: '10px',
              bgcolor: '#0F172A', // Dark Slate matching Image 2
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}>
              <DescriptionIcon sx={{ color: '#FFFFFF', fontSize: 22 }} />
            </Box>
            <Box>
              <Typography variant="h6" fontWeight="800" sx={{ letterSpacing: '-0.02em', color: 'text.primary' }}>
                IntelliDoc
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.primary', opacity: 0.9, fontWeight: 700, display: 'block', mt: -0.5 }}>
                STUDY SYSTEM
              </Typography>
            </Box>
          </Stack>
          {isMobile && (
            <IconButton onClick={() => setDrawerOpen(false)} size="small">
              <CloseIcon />
            </IconButton>
          )}
        </Stack>
      </Box>

      {/* Multi-Document Chat Toggle */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <FormControlLabel
          control={
            <Switch
              checked={chatWithAllDocs}
              onChange={(e) => {
                setChatWithAllDocs(e.target.checked);
                if (e.target.checked) {
                  setCurrentView('chat');
                  showSnackbar('Multi-document chat enabled! Ask questions across all documents.', 'info');
                }
              }}
              color="primary"
            />
          }
          label={
            <Box>
              <Typography variant="body2" fontWeight={600}>
                Chat with All Documents
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {chatWithAllDocs ? 'Searching across all PDFs' : 'Single document mode'}
              </Typography>
            </Box>
          }
        />
      </Box>

      {/* Navigation */}
      <Box sx={{ p: 2 }}>
        <Stack spacing={1}>
          {[
            { id: 'chat', label: 'Chat', icon: <ChatIcon />, disabled: !selectedDocument && !chatWithAllDocs },
            { id: 'documents', label: 'Documents', icon: <DescriptionIcon />, disabled: false },
            { id: 'analytics', label: 'Analytics', icon: <AnalyticsIcon />, disabled: !selectedDocument }
          ].map((item) => (
            <Button
              key={item.id}
              variant={currentView === item.id ? 'contained' : 'text'}
              onClick={() => {
                if (item.id === 'analytics') {
                  selectedDocument && fetchAnalytics(selectedDocument.document_id);
                } else {
                  setCurrentView(item.id as any);
                }
              }}
              startIcon={item.icon}
              fullWidth
              disabled={item.disabled}
              sx={{
                justifyContent: 'flex-start',
                py: 1.2,
                px: 2,
                borderRadius: '8px',
                color: currentView === item.id ? 'primary.contrastText' : 'text.secondary',
                bgcolor: currentView === item.id ? 'primary.main' : 'transparent',
                '&:hover': {
                  bgcolor: currentView === item.id ? 'primary.main' : (darkMode ? 'rgba(255,255,255,0.05)' : '#F1F5F9'),
                  color: 'text.primary',
                },
                '& .MuiButton-startIcon': {
                  color: currentView === item.id ? 'primary.contrastText' : (darkMode ? '#94A3B8' : '#64748B'),
                }
              }}
            >
              {item.label}
            </Button>
          ))}
        </Stack>
      </Box>

      {/* Active Document */}
      {selectedDocument && (
        <Box sx={{ px: 2, pb: 2 }}>
          <Paper sx={{
            p: 2,
            bgcolor: 'action.hover',
          }}>
            <Stack spacing={1}>
              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                Active Document
              </Typography>
              <Typography variant="body2" fontWeight="bold" noWrap>
                {selectedDocument.document_name}
              </Typography>
              <Stack direction="row" spacing={1}>
                <Chip
                  size="small"
                  label={`${selectedDocument.pages} pages`}
                  color="primary"
                />
              </Stack>
            </Stack>
          </Paper>
        </Box>
      )}

      {/* Document List */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ px: 2, pb: 1 }}>
          <Typography variant="caption" color="text.primary" fontWeight={800} sx={{ opacity: 0.8 }}>
            Recent Documents ({documents.length})
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1, overflow: 'auto', px: 2 }}>
          {documents.map((doc) => (
            <Paper
              key={doc.document_id}
              sx={{
                mb: 1,
                p: 1.5,
                cursor: 'pointer',
                bgcolor: selectedDocument?.document_id === doc.document_id
                  ? 'action.selected'
                  : 'background.paper',
                '&:hover': {
                  bgcolor: 'action.hover',
                },
              }}
              onClick={() => {
                setSelectedDocument(doc);
                setCurrentView('chat');
                if (isMobile) setDrawerOpen(false);
              }}
            >
              <Typography variant="body2" fontWeight={600} noWrap sx={{ mb: 0.5 }}>
                {doc.document_name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {doc.pages} pages
              </Typography>
            </Paper>
          ))}
          {documents.length === 0 && (
            <Box sx={{
              textAlign: 'center',
              py: 4,
              px: 2,
            }}>
              <Typography variant="body2" color="text.secondary">
                No documents yet
              </Typography>
            </Box>
          )}
        </Box>
      </Box>

      {/* Bottom Actions */}
      <Box sx={{
        p: 2,
        borderTop: 1,
        borderColor: 'divider',
      }}>
        <Stack spacing={1}>
          <Button
            fullWidth
            variant="contained"
            startIcon={<CloudUploadIcon />}
            onClick={() => setUploadDialog(true)}
          >
            Upload
          </Button>
          <IconButton
            onClick={() => setSettingsDialog(true)}
            sx={{ alignSelf: 'center' }}
          >
            <SettingsIcon />
          </IconButton>
        </Stack>
      </Box>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
        {/* Sidebar */}
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              width: 280,
              boxSizing: 'border-box',
            },
          }}
        >
          <SidebarContent />
        </Drawer>

        {/* Permanent Sidebar for Desktop */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              width: 280,
              boxSizing: 'border-box',
              position: 'relative',
              border: 'none',
            },
          }}
          open
        >
          <SidebarContent />
        </Drawer>

        {/* Main Content */}
        <Box sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
          bgcolor: 'background.default'
        }}>
          {/* Top Bar */}
          <Box sx={{
            borderBottom: 1,
            borderColor: 'divider',
            bgcolor: 'background.paper'
          }}>
            <Toolbar sx={{ minHeight: 64, px: { xs: 2, sm: 3 } }}>
              <IconButton
                edge="start"
                onClick={() => setDrawerOpen(!drawerOpen)}
                sx={{ mr: 2, display: { xs: 'block', md: 'none' }, color: 'text.secondary' }}
              >
                <MenuIcon />
              </IconButton>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h5" fontWeight="800" sx={{ color: 'text.primary', letterSpacing: '-0.02em' }}>
                  {currentView.charAt(0).toUpperCase() + currentView.slice(1)}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                  Management Dashboard
                </Typography>
              </Box>
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Box sx={{ 
                  display: { xs: 'none', sm: 'flex' }, 
                  alignItems: 'center', 
                  bgcolor: '#F1F5F9', 
                  px: 2, 
                  py: 0.5, 
                  borderRadius: '20px',
                  border: '1px solid #E2E8F0'
                }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                    {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </Typography>
                </Box>
                <IconButton
                  onClick={() => setDarkMode(!darkMode)}
                  sx={{ 
                    bgcolor: darkMode ? 'rgba(255,255,255,0.05)' : '#F1F5F9',
                    color: 'text.secondary',
                    '&:hover': { bgcolor: darkMode ? 'rgba(255,255,255,0.1)' : '#E2E8F0' }
                  }}
                >
                  {darkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
                </IconButton>
                <IconButton
                  onClick={() => setSearchOpen(true)}
                  disabled={!selectedDocument}
                  sx={{ 
                    bgcolor: darkMode ? 'rgba(255,255,255,0.05)' : '#F1F5F9',
                    color: 'text.secondary',
                    '&:hover': { bgcolor: darkMode ? 'rgba(255,255,255,0.1)' : '#E2E8F0' }
                  }}
                >
                  <SearchIcon fontSize="small" />
                </IconButton>
                <Avatar 
                  sx={{ 
                    width: 36, 
                    height: 36, 
                    bgcolor: 'primary.main', 
                    color: 'primary.contrastText',
                    fontSize: '0.875rem',
                    fontWeight: 700,
                    border: '2px solid #FFFFFF',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                  }}
                >
                  JD
                </Avatar>
              </Stack>
            </Toolbar>
          </Box>

          {/* Main Content Area */}
          <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
            {currentView === 'chat' && (selectedDocument || chatWithAllDocs) && (
              <ChatInterface
                selectedDocument={selectedDocument}
                messages={messages}
                inputMessage={inputMessage}
                isTyping={isTyping}
                isRecording={isRecording}
                isPlaying={isPlaying}
                darkMode={darkMode}
                messagesEndRef={messagesEndRef}
                fileInputRef={fileInputRef}
                setMessages={setMessages}
                setInputMessage={setInputMessage}
                handleSendMessage={handleSendMessage}
                toggleRecording={() => { }}
                toggleAudioPlayback={() => { }}
                bookmarkMessage={() => { }}
              />
            )}
            {currentView === 'documents' && (
              <DocumentsView
                documents={documents}
                selectedDocument={selectedDocument}
                isDragActive={isDragActive}
                getRootProps={getRootProps}
                getInputProps={getInputProps}
                setSelectedDocument={setSelectedDocument}
                setCurrentView={setCurrentView}
                setUploadDialog={setUploadDialog}
                deleteDocument={deleteDocument}
                fetchAnalytics={fetchAnalytics}
              />
            )}
            {currentView === 'analytics' && (
              <AnalyticsView
                analytics={analytics}
                selectedDocument={selectedDocument}
                setCurrentView={setCurrentView}
              />
            )}
            {!selectedDocument && currentView !== 'documents' && (
              <Box sx={{
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3,
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
              }}>
                <Stack spacing={3} alignItems="center" textAlign="center">
                  <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.contrastText' }} />
                  <Typography variant="h6">
                    No Document Selected
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Upload a document to get started
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<CloudUploadIcon />}
                    onClick={() => setUploadDialog(true)}
                  >
                    Upload Document
                  </Button>
                </Stack>
              </Box>
            )}
          </Box>
        </Box>

        {/* Upload Dialog */}
        <Dialog
          open={uploadDialog}
          onClose={() => setUploadDialog(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Upload Documents</DialogTitle>
          <DialogContent>
            <Paper
              {...getRootProps()}
              sx={{
                p: 4,
                mt: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
                cursor: 'pointer',
                borderStyle: 'dashed',
                borderWidth: 2,
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  backgroundColor: 'action.hover',
                  borderColor: 'primary.main',
                },
              }}
            >
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 48, color: (theme) => theme.palette.primary.main, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                {isDragActive ? 'Drop files here' : 'Choose files or drag and drop'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Supported formats: PDF, TXT, DOC, DOCX
              </Typography>
            </Paper>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setUploadDialog(false)}>Cancel</Button>
            <Button onClick={() => fileInputRef.current?.click()} variant="contained">
              Browse Files
            </Button>
          </DialogActions>
        </Dialog>

        {/* Search Dialog */}
        <Dialog
          open={searchOpen}
          onClose={() => setSearchOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Search Document</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Enter keywords (comma-separated)"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              sx={{ mt: 2 }}
              InputProps={{
                endAdornment: (
                  <IconButton onClick={handleSearch} disabled={loading}>
                    <SearchIcon />
                  </IconButton>
                )
              }}
            />

            {searchResults.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Search Results ({searchResults.length})
                </Typography>
                {searchResults.map((result, index) => (
                  <Paper key={index} sx={{ p: 2, mt: 2 }}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle2" color="primary">
                        Result {index + 1}
                      </Typography>
                      <Chip
                        size="small"
                        label={`Score: ${result.keyword_score?.toFixed(2)}`}
                        color="default"
                        variant="outlined"
                      />
                    </Stack>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      {result.chunk_text}
                    </Typography>
                    {result.matched_keywords && (
                      <Stack direction="row" spacing={1}>
                        {result.matched_keywords.map((keyword: string, i: number) => (
                          <Chip key={i} size="small" label={keyword} />
                        ))}
                      </Stack>
                    )}
                  </Paper>
                ))}
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSearchOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Settings Dialog */}
        <Dialog
          open={settingsDialog}
          onClose={() => setSettingsDialog(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Settings</DialogTitle>
          <DialogContent>
            <Stack spacing={3} sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={darkMode}
                    onChange={(e) => setDarkMode(e.target.checked)}
                  />
                }
                label="Dark Mode"
              />
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Language
                </Typography>
                <FormControl fullWidth>
                  <Select value="English" disabled>
                    <MenuItem value="English">English</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Audio Responses
                </Typography>
                <FormControlLabel
                  control={<Switch defaultChecked />}
                  label="Enable text-to-speech"
                />
              </Box>
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSettingsDialog(false)}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          multiple
          accept=".pdf,.txt,.doc,.docx"
          onChange={handleFileUpload}
        />

        {/* Snackbar */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          <Alert severity={snackbar.severity} sx={{ width: '100%' }}>
            {snackbar.message}
          </Alert>
        </Snackbar>

        {/* Loading Overlay */}
        {loading && (
          <Box
            sx={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              zIndex: 9999,
            }}
          >
            <Paper sx={{ p: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <LinearProgress sx={{ width: 200, mb: 2 }} />
              <Typography variant="h6">Processing...</Typography>
              <Typography variant="body2" color="text.secondary">
                Please wait while we process your request
              </Typography>
            </Paper>
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
};

export default App;