import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders IntelliDoc header', async () => {
  // Mock ResizeObserver
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  // Mock Canvas
  HTMLCanvasElement.prototype.getContext = jest.fn();

  // Mock matchMedia
  window.matchMedia = window.matchMedia || function() {
    return {
      matches: false,
      addListener: function() {},
      removeListener: function() {}
    };
  };

  // Mock fetch
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ documents: [] }),
    })
  ) as jest.Mock;

  render(<App />);
  const linkElement = await screen.findByText(/IntelliDoc/i);
  expect(linkElement).toBeInTheDocument();
});
