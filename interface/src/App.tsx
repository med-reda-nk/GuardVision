import React, { useEffect } from 'react';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';

function App() {
  // Set dark mode by default for this security application
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <Layout>
      <Dashboard />
    </Layout>
  );
}

export default App;