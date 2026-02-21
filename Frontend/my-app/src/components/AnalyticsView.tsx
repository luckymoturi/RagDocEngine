import React from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

interface Analytics {
  basic_info: {
    pages?: number;
    total_words?: number;
    total_chunks?: number;
    document_name?: string;
  };
  content_analysis?: any;
  readability?: {
    level?: string;
    score?: number;
    avg_words_per_sentence?: number;
    avg_chars_per_word?: number;
  };
  key_statistics?: {
    numbers_found?: number;
    percentages_found?: number;
    sample_numbers?: string[];
    sample_percentages?: string[];
  };
  topics?: string[];
  sentiment?: {
    sentiment?: string;
    tone?: string;
    confidence?: number;
  };
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

interface AnalyticsViewProps {
  analytics: Analytics | null;
  selectedDocument: Document | null;
  setCurrentView: (view: 'chat' | 'documents' | 'analytics') => void;
}

const AnalyticsView: React.FC<AnalyticsViewProps> = ({
  analytics,
  selectedDocument,
  setCurrentView
}) => {
  if (!analytics) {
    return (
      <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3 }}>
        <Stack spacing={3} alignItems="center" textAlign="center">
          <AnalyticsIcon sx={{ fontSize: 60, color: 'text.secondary' }} />
          <Typography variant="h5">
            No Analytics Available
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Select a document and generate analytics
          </Typography>
          <Button
            variant="contained"
            startIcon={<ChatIcon />}
            onClick={() => setCurrentView('chat')}
          >
            Go to Chat
          </Button>
        </Stack>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', overflow: 'auto', p: 3 }}>
      <Container maxWidth="lg">
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            startIcon={<ChatIcon />}
            onClick={() => setCurrentView('chat')}
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
            Back to Chat
          </Button>
        </Box>

        <Box sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
          gap: 2,
          mb: 3
        }}>
          {/* Document Overview */}
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Document Overview
              </Typography>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="h4" fontWeight="bold">
                    {analytics.basic_info?.pages || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Pages
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="h4" fontWeight="bold">
                    {Math.round((analytics.basic_info?.total_words || 0) / 1000)}K
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Words
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>

          {/* Readability */}
          {analytics.readability && (
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Readability
                </Typography>
                <Stack spacing={2}>
                  <Box>
                    <Typography variant="h4" fontWeight="bold">
                      {analytics.readability.score || 0}
                    </Typography>
                    <Chip
                      label={analytics.readability.level || 'N/A'}
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min((analytics.readability.score || 0), 100)}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Stack>
              </CardContent>
            </Card>
          )}

          {/* Sentiment */}
          {analytics.sentiment && (
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Sentiment
                </Typography>
                <Stack spacing={2}>
                  <Box>
                    <Typography variant="h5" fontWeight="bold">
                      {analytics.sentiment.sentiment || 'Neutral'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {analytics.sentiment.tone || 'Unknown'}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Confidence: {analytics.sentiment.confidence || 5}/10
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={(analytics.sentiment.confidence || 5) * 10}
                      sx={{ height: 6, borderRadius: 3, mt: 0.5 }}
                    />
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          )}
        </Box>

        {/* Topics */}
        {analytics.topics && Array.isArray(analytics.topics) && analytics.topics.length > 0 ? (
          <Card sx={{ mb: 3, borderRadius: '12px', border: '1px solid', borderColor: 'divider' }}>
            <CardContent sx={{ pb: '16px !important' }}>
              <Typography variant="body2" fontWeight={650} color="#0F172A" gutterBottom>
                Key Topics
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                {analytics.topics.map((topic: any, index: number) => (
                  <Box
                    key={index}
                    sx={{
                      display: 'inline-block',
                      px: 2,
                      py: 0.75,
                      borderRadius: '20px',
                      bgcolor: '#E7EFF3',
                      color: '#0F172A',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      transition: 'all 0.2s ease',
                      '&:hover': { bgcolor: '#D7E5ED', transform: 'translateY(-1px)' },
                    }}
                  >
                    {topic}
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        ) : null}

        {/* Statistics */}
        {analytics.key_statistics && (
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Key Statistics
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2, mt: 2 }}>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                  <Typography variant="h5" fontWeight="bold">
                    {analytics.key_statistics.numbers_found || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Numbers Found
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                  <Typography variant="h5" fontWeight="bold">
                    {analytics.key_statistics.percentages_found || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Percentages Found
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Content Analysis */}
        {analytics.content_analysis ? (
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Content Analysis
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'action.hover', mt: 2 }}>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {typeof analytics.content_analysis === 'string'
                    ? analytics.content_analysis
                    : JSON.stringify(analytics.content_analysis, null, 2)
                  }
                </Typography>
              </Paper>
            </CardContent>
          </Card>
        ) : null}
      </Container>
    </Box>
  );
};

export default AnalyticsView;
