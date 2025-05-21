import React, { useState } from 'react';
import { 
  mockAlerts, mockCameras, mockCrowdData, mockSecurityStats 
} from '../utils/mock-data';
import VideoFeed from '../components/dashboard/VideoFeed';
import DetectionAlerts from '../components/dashboard/DetectionAlerts';
import CrowdDensityChart from '../components/dashboard/CrowdDensityChart';
import StatisticsCard from '../components/dashboard/StatisticsCard';
import CameraModal from '../components/modals/CameraModal';
import { Camera } from '../types';
import { Upload, FileType } from 'lucide-react';

const Dashboard: React.FC = () => {
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const handleMaximize = (camera: Camera) => {
    setSelectedCamera(camera);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    // Handle model file upload here
  };
  
  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Security Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">Real-time monitoring and threat detection</p>
      </div>
      
      {/* Stats Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {mockSecurityStats.map((stat, index) => (
          <StatisticsCard key={index} stat={stat} />
        ))}
      </div>
      
      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Camera Grid */}
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Live Camera Feeds</h2>
              
              {/* Model Upload Zone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`
                  border-2 border-dashed rounded-lg p-2 flex items-center space-x-2
                  transition-colors duration-200 ease-in-out cursor-pointer
                  ${isDragging
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-300 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500'
                  }
                `}
              >
                <Upload className="h-5 w-5 text-gray-400 dark:text-gray-600" />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Drop AI model here
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {mockCameras.slice(0, 4).map((camera) => (
                <VideoFeed 
                  key={camera.id} 
                  camera={camera} 
                  onMaximize={handleMaximize}
                />
              ))}
            </div>
          </div>
          
          {/* Crowd Density Chart */}
          <CrowdDensityChart 
            data={mockCrowdData} 
            location="Main Lobby" 
          />
        </div>
        
        {/* Alert Section */}
        <div className="lg:col-span-1">
          <DetectionAlerts alerts={mockAlerts} />
        </div>
      </div>
      
      {/* Camera Modal */}
      {selectedCamera && (
        <CameraModal 
          camera={selectedCamera} 
          onClose={() => setSelectedCamera(null)} 
        />
      )}
    </div>
  );
};

export default Dashboard;