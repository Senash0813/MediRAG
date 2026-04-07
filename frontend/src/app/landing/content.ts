/** Static copy for the marketing landing page */

export const landingPipelines = [
  {
    name: 'Neurosciences',
    description:
      'Specialized coverage for disorders of the brain, spinal cord, and peripheral nervous system—grounded in neurology and neurosurgical sources.',
    ragFocus:
      'Retrieval-augmented answers tuned to neuroscience corpora, so responses stay tied to curated clinical material.',
  },
  {
    name: 'Cardiovascular',
    description:
      'Heart and vascular care contexts—from cardiology through cardiothoracic and vascular surgery—organized for precise literature-backed replies.',
    ragFocus:
      'A dedicated pipeline that routes queries to cardiovascular indexes and merges evidence before generation.',
  },
  {
    name: 'Internal Medicine',
    description:
      'Broad internal medicine and related specialties (renal, endocrine, pulmonary, GI, hematology, and more) in one structured knowledge path.',
    ragFocus:
      'Multi-domain retrieval with parameters suited to longer-form, cross-specialty clinical questions.',
  },
  {
    name: 'Primary Care & Mental Health',
    description:
      'Holistic primary care, psychiatry, psychology, nursing, family medicine, and geriatrics—with an emphasis on safe, transparent answering.',
    ragFocus:
      'Includes verification-oriented output when supported by the backend, surfacing limitations alongside synthesized guidance.',
  },
] as const;

export const landingTestimonials = [
  {
    quote:
      'The split pipelines mirror how our department actually thinks—neuro vs cards vs medicine—instead of one generic assistant.',
    name: 'Dr. A. Chen',
    role: 'Attending neurologist',
    org: 'Academic medical center',
  },
  {
    quote:
      'We piloted it with early users in quality; the ability to see which domain you are querying reduced misuse and improved trust.',
    name: 'M. Okonkwo',
    role: 'Clinical informatics lead',
    org: 'Regional health system',
  },
  {
    quote:
      'Primary care mode’s structured limitations field is what we needed for resident teaching—not just an answer, but where it’s thin.',
    name: 'Dr. S. Patel',
    role: 'Program director, family medicine',
    org: 'Community hospital',
  },
  {
    quote:
      'Fast iteration on four backends is rare in med RAG demos; this feels closer to something you could actually evaluate in production.',
    name: 'J. Rivera',
    role: 'Product lead, digital health',
    org: 'Industry partner',
  },
  {
    quote:
      'Cardiovascular queries pulled from the right index meant fewer “close enough” answers during case review—we could trace claims back to the literature set.',
    name: 'Dr. E. Matsumoto',
    role: 'Interventional cardiologist',
    org: 'Urban tertiary center',
  },
  {
    quote:
      'Our nurses used internal-medicine routing for complex med-surg questions; the interface made it obvious when to escalate to a specialist pipeline.',
    name: 'K. Osei',
    role: 'Chief nursing informatics officer',
    org: 'Integrated delivery network',
  },
  {
    quote:
      'For grant prep we needed reproducible, domain-scoped retrieval—not a black box. The pipeline labels matched how we section our IRB and methods text.',
    name: 'Dr. R. Feldman',
    role: 'Health services researcher',
    org: 'Public university',
  },
] as const;
