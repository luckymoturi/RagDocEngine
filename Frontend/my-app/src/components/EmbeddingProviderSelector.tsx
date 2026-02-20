import React, { useState, useEffect } from 'react';
import API_URL from '../config';
import {
  Box,
  FormControl,
  FormLabel,
  RadioGroup,
  Radio,
  Typography,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';
import axios from 'axios';

interface EmbeddingProviderSelectorProps {
  darkMode?: boolean;
}

const EmbeddingProviderSelector: React.FC<EmbeddingProviderSelectorProps> = ({ darkMode = false }) => {
  const [provider, setProvider] = useState<string>('gemini');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [availableProviders, setAvailableProviders] = useState<{
    gemini_available: boolean;
    azure_openai_available: boolean;
  }>({
    gemini_available: false,
    azure_openai_available: false
  });

  // Fetch current provider on mount
  useEffect(() => {
    fetchCurrentProvider();
  }, []);

  const fetchCurrentProvider = async () => {
    try {
      const response = await axios.get(`${API_URL}/embedding-provider`);
      setProvider(response.data.provider);
      setAvailableProviders({
        gemini_available: response.data.gemini_available,
        azure_openai_available: response.data.azure_openai_available
      });
    } catch (err) {
      console.error('Error fetching provider:', err);
      setError('Failed to fetch current provider');
    }
  };

  const handleProviderChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const newProvider = event.target.value;
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await axios.post(`${API_URL}/embedding-provider`, {
        provider: newProvider
      });
      
      setProvider(newProvider);
      setSuccess(`Successfully switched to ${newProvider === 'gemini' ? 'Gemini' : 'Azure OpenAI'}`);
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(''), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to change provider');
      // Revert to previous provider
      fetchCurrentProvider();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 2,
        bgcolor: darkMode ? 'grey.900' : 'grey.50',
        border: 1,
        borderColor: darkMode ? 'grey.700' : 'grey.300'
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        🤖 Embedding Model
        {loading && <CircularProgress size={20} />}
      </Typography>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Select which AI model to use for document embeddings and search
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess('')}>
          {success}
        </Alert>
      )}

      <FormControl component="fieldset" disabled={loading}>
        <RadioGroup value={provider} onChange={handleProviderChange}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {/* Gemini Option */}
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: 1,
                borderColor: provider === 'gemini' ? 'primary.main' : 'grey.300',
                bgcolor: provider === 'gemini' ? (darkMode ? 'primary.dark' : 'primary.light') : 'transparent',
                cursor: 'pointer',
                transition: 'all 0.2s',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: darkMode ? 'grey.800' : 'grey.100'
                }
              }}
              onClick={() => !loading && handleProviderChange({ target: { value: 'gemini' } } as any)}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Radio value="gemini" disabled={!availableProviders.gemini_available} />
                  <Box>
                    <Typography variant="subtitle1" fontWeight="bold">
                      Google Gemini
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      3072-dimensional embeddings • High quality • Fast
                    </Typography>
                  </Box>
                </Box>
                {availableProviders.gemini_available ? (
                  <Chip label="Available" color="success" size="small" />
                ) : (
                  <Chip label="Not Configured" color="error" size="small" />
                )}
              </Box>
            </Box>

            {/* Azure OpenAI Option */}
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: 1,
                borderColor: provider === 'azure_openai' ? 'primary.main' : 'grey.300',
                bgcolor: provider === 'azure_openai' ? (darkMode ? 'primary.dark' : 'primary.light') : 'transparent',
                cursor: 'pointer',
                transition: 'all 0.2s',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: darkMode ? 'grey.800' : 'grey.100'
                }
              }}
              onClick={() => !loading && handleProviderChange({ target: { value: 'azure_openai' } } as any)}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Radio value="azure_openai" disabled={!availableProviders.azure_openai_available} />
                  <Box>
                    <Typography variant="subtitle1" fontWeight="bold">
                      Azure OpenAI
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      1536-dimensional embeddings • Enterprise-grade • Reliable
                    </Typography>
                  </Box>
                </Box>
                {availableProviders.azure_openai_available ? (
                  <Chip label="Available" color="success" size="small" />
                ) : (
                  <Chip label="Not Configured" color="error" size="small" />
                )}
              </Box>
            </Box>
          </Box>
        </RadioGroup>
      </FormControl>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
        💡 Tip: Both models provide high-quality embeddings. Choose based on your API availability.
      </Typography>
    </Box>
  );
};

export default EmbeddingProviderSelector;
