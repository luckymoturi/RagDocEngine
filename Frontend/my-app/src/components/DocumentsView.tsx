import React from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  IconButton,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Description as DescriptionIcon,
  Delete as DeleteIcon,
  Analytics as AnalyticsIcon,
  Chat as ChatIcon,
} from '@mui/icons-material';

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

interface DocumentsViewProps {
  documents: Document[];
  selectedDocument: Document | null;
  isDragActive: boolean;
  getRootProps: () => any;
  getInputProps: () => any;
  setSelectedDocument: (doc: Document) => void;
  setCurrentView: (view: 'chat' | 'documents' | 'analytics') => void;
  setUploadDialog: (open: boolean) => void;
  deleteDocument: (documentId: string) => void;
  fetchAnalytics: (documentId: string) => void;
}

const DocumentsView: React.FC<DocumentsViewProps> = ({
  documents,
  selectedDocument,
  isDragActive,
  getRootProps,
  getInputProps,
  setSelectedDocument,
  setCurrentView,
  setUploadDialog,
  deleteDocument,
  fetchAnalytics
}) => {
  return (
    <Box sx={{ height: '100%', overflow: 'auto', p: 3 }}>
      <Container maxWidth="lg">
        {documents.length === 0 ? (
          <Paper
            {...getRootProps()}
            sx={{
              p: 6,
              textAlign: 'center',
              minHeight: '400px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              cursor: 'pointer',
              borderStyle: 'dashed',
              borderWidth: 2,
              borderColor: isDragActive ? 'primary.main' : 'divider',
              bgcolor: isDragActive ? 'action.hover' : 'background.paper',
              '&:hover': {
                bgcolor: 'action.hover',
                borderColor: 'primary.main',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              Upload Your First Document
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {isDragActive
                ? 'Drop the files here...'
                : 'Drag and drop files here or click to browse'
              }
            </Typography>
            <Stack direction="row" spacing={1}>
              {['PDF', 'TXT', 'DOC', 'DOCX'].map((format) => (
                <Chip key={format} label={format} size="small" />
              ))}
            </Stack>
          </Paper>
        ) : (
          <>
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                onClick={() => setUploadDialog(true)}
                sx={{
                  bgcolor: '#BFC9D1',
                  color: '#0F172A',
                  fontWeight: 600,
                  '&:hover': {
                    bgcolor: '#A8B4BD',
                  }
                }}
              >
                Upload
              </Button>
            </Box>

            <Box sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
              gap: 2
            }}>
              {documents.map((doc) => (
                <Card
                  key={doc.document_id}
                  sx={{
                    cursor: 'pointer',
                    borderRadius: '12px',
                    border: '1px solid',
                    borderColor: 'divider',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
                    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
                      borderColor: 'primary.main',
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                        <Stack direction="row" spacing={1.5} flex={1} minWidth={0}>
                          <Box
                            sx={{
                              width: 44,
                              height: 44,
                              borderRadius: '10px',
                              bgcolor: '#F1F5F9', // Light Slate-100 like Image 2 icons
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              flexShrink: 0,
                            }}
                          >
                            <DescriptionIcon sx={{ color: '#64748B', fontSize: 24 }} />
                          </Box>
                          <Box flex={1} minWidth={0}>
                            <Typography variant="body1" fontWeight={600} noWrap>
                              {doc.document_name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {doc.pages} pages
                            </Typography>
                          </Box>
                        </Stack>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteDocument(doc.document_id);
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Stack>

                      <Stack direction="row" spacing={1.5}>
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<ChatIcon />}
                          onClick={() => {
                            setSelectedDocument(doc);
                            setCurrentView('chat');
                          }}
                          disabled={doc.status === 'processing' || doc.status === 'error'}
                          fullWidth
                          sx={{
                            bgcolor: '#BFC9D1',
                            color: '#0F172A',
                            fontWeight: 600,
                            '&:hover': {
                              bgcolor: '#A8B4BD',
                            }
                          }}
                        >
                          Chat
                        </Button>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<AnalyticsIcon />}
                          onClick={() => {
                            setSelectedDocument(doc);
                            fetchAnalytics(doc.document_id);
                          }}
                          disabled={doc.status === 'processing' || doc.status === 'error'}
                          fullWidth
                          sx={{
                            borderColor: 'divider',
                            color: 'text.secondary',
                            '&:hover': {
                              borderColor: 'text.primary',
                              bgcolor: '#F8FAFC',
                              color: 'text.primary',
                            }
                          }}
                        >
                          Analytics
                        </Button>
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </>
        )}
      </Container>
    </Box>
  );
};

export default DocumentsView;
