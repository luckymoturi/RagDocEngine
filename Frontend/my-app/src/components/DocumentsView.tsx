import React, { useState } from 'react';
import API_URL from '../config';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Description as DescriptionIcon,
  Delete as DeleteIcon,
  Analytics as AnalyticsIcon,
  Chat as ChatIcon,
  Visibility as VisibilityIcon,
  Close as CloseIcon,
  OpenInNew as OpenInNewIcon,
  Download as DownloadIcon,
  CalendarToday as CalendarIcon,
  Article as ArticleIcon,
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

const statusColor: Record<string, string> = {
  ready: '#22c55e',
  processing: '#f59e0b',
  error: '#ef4444',
};

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
  fetchAnalytics,
}) => {
  const [pdfViewer, setPdfViewer] = useState<{ open: boolean; doc: Document | null }>({
    open: false,
    doc: null,
  });
  const [pdfLoading, setPdfLoading] = useState(false);

  const openPdfViewer = (doc: Document) => {
    setPdfViewer({ open: true, doc });
    setPdfLoading(true);
  };

  const closePdfViewer = () => {
    setPdfViewer({ open: false, doc: null });
    setPdfLoading(false);
  };

  const pdfUrl = pdfViewer.doc
    ? `${API_URL}/document/${pdfViewer.doc.document_id}/pdf`
    : '';

  const formatDate = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <Box sx={{ height: '100%', overflow: 'auto', p: 3 }}>
      <Container maxWidth="lg">

        {documents.length === 0 ? (
          /* â”€â”€ Empty State â”€â”€ */
          <Paper
            {...getRootProps()}
            sx={{
              p: 8,
              textAlign: 'center',
              minHeight: 420,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              cursor: 'pointer',
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'divider',
              bgcolor: isDragActive ? 'action.hover' : 'background.paper',
              transition: 'all 0.2s ease',
              '&:hover': { bgcolor: 'action.hover', borderColor: 'primary.main' },
            }}
          >
            <input {...getInputProps()} />
            <Box sx={{
              width: 80, height: 80, borderRadius: '20px', bgcolor: '#F1F5F9',
              display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 3,
            }}>
              <CloudUploadIcon sx={{ fontSize: 40, color: '#64748B' }} />
            </Box>
            <Typography variant="h5" fontWeight={700} gutterBottom>
              {isDragActive ? 'Drop files here...' : 'Upload Your First Document'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3, maxWidth: 340 }}>
              Drag and drop your files here or click to browse. Supports PDF, TXT, DOC, DOCX.
            </Typography>
            <Stack direction="row" spacing={1}>
              {['PDF', 'TXT', 'DOC', 'DOCX'].map((f) => (
                <Chip key={f} label={f} size="small" variant="outlined" />
              ))}
            </Stack>
          </Paper>
        ) : (
          <>
            {/* â”€â”€ Header â”€â”€ */}
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 3 }}>
              <Box>
                <Typography variant="h5" fontWeight={700}>Documents</Typography>
                <Typography variant="body2" color="text.secondary">
                  {documents.length} document{documents.length !== 1 ? 's' : ''} in your library
                </Typography>
              </Box>
              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                onClick={() => setUploadDialog(true)}
                sx={{
                  bgcolor: '#0F172A', color: '#fff', fontWeight: 600, px: 2.5,
                  '&:hover': { bgcolor: '#1E293B' },
                }}
              >
                Upload
              </Button>
            </Stack>

            {/* â”€â”€ Document Grid â”€â”€ */}
            <Box sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
              gap: 2,
            }}>
              {documents.map((doc) => (
                <Card
                  key={doc.document_id}
                  sx={{
                    borderRadius: '14px',
                    border: '1.5px solid',
                    borderColor: selectedDocument?.document_id === doc.document_id ? '#0F172A' : 'divider',
                    boxShadow: selectedDocument?.document_id === doc.document_id
                      ? '0 0 0 2px rgba(15,23,42,0.08)' : '0 1px 3px rgba(0,0,0,0.08)',
                    transition: 'all 0.2s ease',
                    '&:hover': { boxShadow: '0 8px 24px rgba(0,0,0,0.1)', transform: 'translateY(-2px)' },
                  }}
                >
                  <CardContent sx={{ p: 2.5, pb: '16px !important' }}>
                    <Stack spacing={2}>

                      {/* Name + Status + Delete */}
                      <Stack direction="row" spacing={1.5} alignItems="flex-start">
                        <Box sx={{
                          width: 46, height: 46, borderRadius: '10px', bgcolor: '#F1F5F9',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                        }}>
                          <DescriptionIcon sx={{ color: '#64748B', fontSize: 24 }} />
                        </Box>
                        <Box flex={1} minWidth={0}>
                          <Typography variant="body1" fontWeight={650} noWrap title={doc.document_name}>
                            {doc.document_name}
                          </Typography>
                          <Stack direction="row" spacing={0.8} alignItems="center" mt={0.3}>
                            <Box sx={{
                              width: 7, height: 7, borderRadius: '50%',
                              bgcolor: statusColor[doc.status || 'ready'],
                            }} />
                            <Typography variant="caption" color="text.secondary" textTransform="capitalize">
                              {doc.status || 'ready'}
                            </Typography>
                          </Stack>
                        </Box>
                        <Tooltip title="Delete document">
                          <IconButton
                            size="small"
                            onClick={(e) => { e.stopPropagation(); deleteDocument(doc.document_id); }}
                            sx={{ color: 'text.secondary', '&:hover': { color: '#ef4444' } }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Stack>

                      {/* Meta info */}
                      <Stack direction="row" spacing={2} sx={{ px: 0.5 }}>
                        <Stack direction="row" spacing={0.5} alignItems="center">
                          <ArticleIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                          <Typography variant="caption" color="text.secondary">{doc.pages} pages</Typography>
                        </Stack>
                        <Stack direction="row" spacing={0.5} alignItems="center">
                          <CalendarIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                          <Typography variant="caption" color="text.secondary">{formatDate(doc.upload_time)}</Typography>
                        </Stack>
                      </Stack>

                      {/* Buttons */}
                      <Stack direction="row" spacing={1}>
                        <Tooltip title="View PDF">
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<VisibilityIcon sx={{ color: 'inherit' }} />}
                            onClick={() => openPdfViewer(doc)}
                            disabled={doc.status !== 'ready'}
                            fullWidth
                            sx={{
                              borderColor: '#CBD5E1', color: '#64748B', fontSize: '0.75rem',
                              '& .MuiButton-startIcon': { color: 'inherit' },
                              '&:hover': { borderColor: '#0F172A', color: '#0F172A', bgcolor: '#F1F5F9',
                                '& .MuiButton-startIcon': { color: '#0F172A' } },
                            }}
                          >
                            View
                          </Button>
                        </Tooltip>

                        <Tooltip title="Chat with document">
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<ChatIcon sx={{ color: 'inherit' }} />}
                            onClick={() => { setSelectedDocument(doc); setCurrentView('chat'); }}
                            disabled={doc.status !== 'ready'}
                            fullWidth
                            sx={{
                              borderColor: '#CBD5E1', color: '#64748B', fontSize: '0.75rem',
                              '& .MuiButton-startIcon': { color: 'inherit' },
                              '&:hover': { borderColor: '#0F172A', color: '#0F172A', bgcolor: '#F1F5F9',
                                '& .MuiButton-startIcon': { color: '#0F172A' } },
                            }}
                          >
                            Chat
                          </Button>
                        </Tooltip>

                        <Tooltip title="Analytics">
                          <IconButton
                            size="small"
                            onClick={() => { setSelectedDocument(doc); fetchAnalytics(doc.document_id); }}
                            disabled={doc.status !== 'ready'}
                            sx={{
                              border: '1px solid', borderColor: 'divider', borderRadius: '6px',
                              color: 'text.secondary', px: 1,
                              '&:hover': { borderColor: '#0F172A', color: '#0F172A' },
                            }}
                          >
                            <AnalyticsIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Stack>

                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </>
        )}
      </Container>

      {/* â”€â”€ PDF Viewer Dialog â”€â”€ */}
      <Dialog
        open={pdfViewer.open}
        onClose={closePdfViewer}
        maxWidth={false}
        fullWidth
        PaperProps={{
          sx: {
            width: '90vw',
            height: '92vh',
            maxWidth: '1100px',
            borderRadius: '16px',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        {/* Header */}
        <DialogTitle sx={{
          px: 3, py: 1.5,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderBottom: '1px solid', borderColor: 'divider',
          bgcolor: 'background.paper', flexShrink: 0,
        }}>
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <Box sx={{
              width: 36, height: 36, borderRadius: '8px', bgcolor: '#F1F5F9',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <DescriptionIcon sx={{ color: '#64748B', fontSize: 20 }} />
            </Box>
            <Box>
              <Typography variant="subtitle1" fontWeight={700} noWrap sx={{ maxWidth: 500 }}>
                {pdfViewer.doc?.document_name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {pdfViewer.doc?.pages} pages
              </Typography>
            </Box>
          </Stack>

          <Stack direction="row" spacing={0.5}>
            <Tooltip title="Open in new tab">
              <IconButton size="small" onClick={() => window.open(pdfUrl, '_blank')} sx={{ color: 'text.secondary' }}>
                <OpenInNewIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download">
              <IconButton
                size="small"
                component="a"
                href={pdfUrl}
                download={pdfViewer.doc?.document_name}
                sx={{ color: 'text.secondary' }}
              >
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Close">
              <IconButton size="small" onClick={closePdfViewer} sx={{ color: 'text.secondary' }}>
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </DialogTitle>

        {pdfLoading && <LinearProgress sx={{ height: 2, flexShrink: 0 }} />}

        {/* PDF iframe */}
        <DialogContent sx={{ p: 0, flex: 1, overflow: 'hidden', bgcolor: '#525659' }}>
          {pdfViewer.open && pdfUrl && (
            <iframe
              src={pdfUrl}
              title={pdfViewer.doc?.document_name || 'PDF Viewer'}
              width="100%"
              height="100%"
              style={{ border: 'none', display: 'block' }}
              onLoad={() => setPdfLoading(false)}
              onError={() => setPdfLoading(false)}
            />
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default DocumentsView;
