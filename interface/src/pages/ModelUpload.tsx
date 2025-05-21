import React, { useState, useCallback } from 'react';
import { Upload, FileType, CheckCircle, AlertCircle } from 'lucide-react';

interface ModelFile {
  name: string;
  size: number;
  status: 'success' | 'error';
  message?: string;
}

const ModelUpload: React.FC = () => {
  const [models, setModels] = useState<ModelFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const validateFile = (file: File): ModelFile => {
    // Check if file is a Python file
    if (!file.name.endsWith('.py') && !file.name.endsWith('.pth') && !file.name.endsWith('.pt')) {
      return {
        name: file.name,
        size: file.size,
        status: 'error',
        message: 'Invalid file type. Only .py, .pth, and .pt files are allowed.'
      };
    }

    return {
      name: file.name,
      size: file.size,
      status: 'success'
    };
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const newModels = files.map(validateFile);
    setModels(prev => [...prev, ...newModels]);
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Management</h1>
        <p className="text-gray-600 dark:text-gray-400">Upload and manage your Python models</p>
      </div>

      {/* Upload Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8
          transition-colors duration-200 ease-in-out
          ${isDragging
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-gray-300 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500'
          }
        `}
      >
        <div className="flex flex-col items-center justify-center text-center">
          <Upload className="h-12 w-12 text-gray-400 dark:text-gray-600 mb-4" />
          <p className="text-xl font-medium text-gray-700 dark:text-gray-300 mb-2">
            Drag and drop your Python models here
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Supports .py, .pth, and .pt files
          </p>
        </div>
      </div>

      {/* Model List */}
      {models.length > 0 && (
        <div className="mt-8">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Uploaded Models
          </h2>
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow overflow-hidden">
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {models.map((model, index) => (
                <div
                  key={`${model.name}-${index}`}
                  className="p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <div className="flex items-center">
                    <div className="mr-3">
                      {model.status === 'success' ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-red-500" />
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {model.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {formatFileSize(model.size)}
                      </p>
                      {model.status === 'error' && (
                        <p className="text-xs text-red-500 mt-1">{model.message}</p>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => setModels(models.filter((_, i) => i !== index))}
                    className="text-sm text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelUpload;