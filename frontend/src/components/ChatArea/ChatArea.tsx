'use client';

import React, { useState } from 'react';
import { Brain, Heart, Stethoscope, Activity } from 'lucide-react';
import { ClusterCard } from './ClusterCard';

export const ChatArea = () => {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);

  const clusters = [
    {
      number: 1,
      name: 'Neurosciences',
      icon: Brain,
      description: 'Specialized care for disorders of the brain, spinal cord, and nervous system.',
      specialties: [
        'Neurology',
        'Neurosurgery',
      ],
    },
    {
      number: 2,
      name: 'Cardiovascular',
      icon: Heart,
      description: 'Comprehensive heart and vascular system care, including surgical interventions.',
      specialties: [
        'Cardiology',
        'Cardiothoracic Surgery',
        'Vascular Surgery',
      ],
    },
    {
      number: 3,
      name: 'Internal Medicine',
      icon: Stethoscope,
      description: 'Broad range of medical specialties covering organ systems and general health.',
      specialties: [
        'General Internal Medicine',
        'General Pediatrics',
        'General Surgery',
        'Nephrology',
        'Endocrinology & Metabolism',
        'Hematology',
        'Pulmonology & Respiratory Medicine',
        'Gastroenterology & Hepatology',
      ],
    },
    {
      number: 4,
      name: 'Primary Care & Mental Health',
      icon: Activity,
      description: 'Holistic care focusing on mental wellness, family health, and aging populations.',
      specialties: [
        'Psychiatry',
        'Psychology & Behavioral Health',
        'Nursing',
        'Family Medicine & Primary Care',
        'Geriatrics',
      ],
    },
  ];

  const handleClusterSelect = (clusterNumber: number) => {
    // Toggle selection - if same cluster is clicked, deselect it
    setSelectedCluster(selectedCluster === clusterNumber ? null : clusterNumber);
    // TODO: Activate the respective pipeline when backend is ready
    if (selectedCluster !== clusterNumber) {
      console.log(`Pipeline ${clusterNumber} activated for Cluster ${clusterNumber}`);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center max-w-6xl mx-auto w-full px-6 pb-32 overflow-y-auto scrollbar-hide">
      <div className="w-full mb-12 animate-in slide-in-from-bottom-4 duration-700 text-center">
        <h1 className="text-6xl font-medium mb-2 tracking-tight">
          <span className="bg-gradient-to-r from-[#4285f4] via-[#9b72cb] to-[#d96570] bg-clip-text text-transparent">
            Welcome to MediRAG
          </span>
        </h1>
        <h4 className="text-3xl font-medium text-[#444746] tracking-tight">
          Find your focus. Ask your questions.
        </h4>
      </div>

      {/* Clusters Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full animate-in fade-in zoom-in-95 duration-1000 delay-200">
        {clusters.map((cluster) => (
          <ClusterCard
            key={cluster.number}
            clusterNumber={cluster.number}
            name={cluster.name}
            icon={cluster.icon}
            description={cluster.description}
            onSelect={() => handleClusterSelect(cluster.number)}
            isSelected={selectedCluster === cluster.number}
          />
        ))}
      </div>
    </div>
  );
};
