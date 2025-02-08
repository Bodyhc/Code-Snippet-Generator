import React, { useState } from "react";
import { Copy, Code2, CheckCircle2, Search, Filter } from "lucide-react";

type Technology = "react" | "nextjs" | "typescript" | "javascript" | "html";
type Snippet = {
  id: string;
  title: string;
  description: string;
  code: string;
  technology: Technology;
  tags?: string[];
  examples?: string;
};

const snippets: Snippet[] = [
  // React Snippets
  {
    id: "react-1",
    title: "Custom Hook - useLocalStorage",
    description: "A custom hook for managing state in localStorage",
    technology: "react",
    tags: ["hooks", "storage"],
    code: `function useLocalStorage<T>(key: string, initialValue: T) {
  // Get from local storage then
  // parse stored json or return initialValue
  const readValue = () => {
    if (typeof window === 'undefined') {
      return initialValue;
    }

    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(\`Error reading localStorage key "\${key}":\`, error);
      return initialValue;
    }
  };

  // State to store our value
  // Pass initial state function to useState so logic is only executed once
  const [storedValue, setStoredValue] = useState<T>(readValue);

  // Return a wrapped version of useState's setter function that ...
  // ... persists the new value to localStorage.
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      // Allow value to be a function so we have same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      // Save state
      setStoredValue(valueToStore);
      // Save to local storage
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.warn(\`Error setting localStorage key "\${key}":\`, error);
    }
  };

  return [storedValue, setValue] as const;
}`,
  },
  {
    id: "react-2",
    title: "Custom Hook - useMediaQuery",
    description: "A custom hook for responsive design with media queries",
    technology: "react",
    tags: ["hooks", "responsive"],
    code: `import { useState, useEffect } from 'react';

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(query);
    if (media.matches !== matches) {
      setMatches(media.matches);
    }
    const listener = () => setMatches(media.matches);
    media.addListener(listener);
    return () => media.removeListener(listener);
  }, [matches, query]);

  return matches;
}`,
  },
  {
    id: "react-3",
    title: "Error Boundary Component",
    description: "A component for handling and displaying errors gracefully",
    technology: "react",
    tags: ["error-handling", "components"],
    code: `import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div role="alert">
          <h2>Something went wrong</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {this.state.error?.toString()}
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}`,
  },
  {
    id: "react-4",
    title: "Form Validation Hook",
    description: "A custom hook for form validation with error handling",
    technology: "react",
    tags: ["hooks", "forms"],
    code: `import { useState, useCallback } from 'react';

interface Validation {
  required?: boolean;
  pattern?: RegExp;
  minLength?: number;
  maxLength?: number;
  custom?: (value: string) => boolean;
}

interface ValidationRules {
  [key: string]: Validation;
}

interface Errors {
  [key: string]: string;
}

function useFormValidation<T extends { [key: string]: string }>(
  initialState: T,
  validationRules: ValidationRules
) {
  const [values, setValues] = useState<T>(initialState);
  const [errors, setErrors] = useState<Errors>({});
  const [isValid, setIsValid] = useState(false);

  const validateField = useCallback(
    (name: string, value: string): string => {
      const rules = validationRules[name];
      if (!rules) return '';

      if (rules.required && !value) {
        return 'This field is required';
      }

      if (rules.pattern && !rules.pattern.test(value)) {
        return 'Invalid format';
      }

      if (rules.minLength && value.length < rules.minLength) {
        return \`Minimum length is \${rules.minLength}\`;
      }

      if (rules.maxLength && value.length > rules.maxLength) {
        return \`Maximum length is \${rules.maxLength}\`;
      }

      if (rules.custom && !rules.custom(value)) {
        return 'Invalid value';
      }

      return '';
    },
    [validationRules]
  );

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const { name, value } = event.target;
      setValues(prev => ({ ...prev, [name]: value }));
      
      const error = validateField(name, value);
      setErrors(prev => ({ ...prev, [name]: error }));
      
      const newErrors = { ...errors, [name]: error };
      setIsValid(
        Object.keys(validationRules).every(
          key => !newErrors[key] && values[key as keyof T]
        )
      );
    },
    [validateField, errors, values, validationRules]
  );

  return { values, errors, isValid, handleChange };
}`,
  },
  {
    id: "react-5",
    title: "Infinite Scroll Hook",
    description: "A custom hook for implementing infinite scroll",
    technology: "react",
    tags: ["hooks", "scroll"],
    code: `import { useState, useEffect, useCallback, useRef } from 'react';

interface UseInfiniteScrollOptions {
  threshold?: number;
  initialPage?: number;
}

function useInfiniteScroll<T>(
  loadMore: (page: number) => Promise<T[]>,
  options: UseInfiniteScrollOptions = {}
) {
  const { threshold = 100, initialPage = 1 } = options;
  
  const [items, setItems] = useState<T[]>([]);
  const [page, setPage] = useState(initialPage);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  const observer = useRef<IntersectionObserver>();
  const lastElementRef = useCallback(
    (node: HTMLElement | null) => {
      if (loading) return;
      
      if (observer.current) {
        observer.current.disconnect();
      }
      
      observer.current = new IntersectionObserver(
        entries => {
          if (entries[0].isIntersecting && hasMore) {
            setPage(prev => prev + 1);
          }
        },
        { rootMargin: \`\${threshold}px\` }
      );
      
      if (node) {
        observer.current.observe(node);
      }
    },
    [loading, hasMore, threshold]
  );
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const newItems = await loadMore(page);
        setItems(prev => [...prev, ...newItems]);
        setHasMore(newItems.length > 0);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('An error occurred'));
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [page, loadMore]);
  
  return {
    items,
    loading,
    error,
    hasMore,
    lastElementRef
  };
}`,
  },

  // Next.js Snippets
  {
    id: "next-1",
    title: "Next.js API Route with Rate Limiting",
    description: "API route implementation with rate limiting",
    technology: "nextjs",
    tags: ["api", "security"],
    code: `import type { NextApiRequest, NextApiResponse } from 'next';
import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.REDIS_URL!,
  token: process.env.REDIS_TOKEN!
});

interface RateLimitConfig {
  limit: number;
  window: number; // in seconds
}

export async function rateLimit(
  req: NextApiRequest,
  config: RateLimitConfig
): Promise<{
  isLimited: boolean;
  remaining: number;
  reset: number;
}> {
  const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress;
  const key = \`rate-limit:\${ip}\`;
  const now = Date.now();
  const window = config.window * 1000;

  const multi = redis
    .multi()
    .zremrangebyscore(key, 0, now - window)
    .zadd(key, now, now.toString())
    .zcard(key)
    .pexpire(key, window);

  const [, , current] = await multi.exec();
  const remaining = config.limit - (current as number);
  const reset = now + window;

  return {
    isLimited: remaining <= 0,
    remaining: Math.max(0, remaining),
    reset
  };
}`,
  },
  {
    id: "next-2",
    title: "Next.js Middleware Authentication",
    description: "Middleware for handling authentication and protected routes",
    technology: "nextjs",
    tags: ["auth", "middleware"],
    code: `import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const protectedPaths = ['/dashboard', '/profile', '/settings'];
const authPaths = ['/login', '/register', '/forgot-password'];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get('auth-token');
  
  const isProtectedPath = protectedPaths.some(path => 
    pathname.startsWith(path)
  );
  
  const isAuthPath = authPaths.some(path => 
    pathname.startsWith(path)
  );

  if (isAuthPath && token) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  if (isProtectedPath && !token) {
    const url = new URL('/login', request.url);
    url.searchParams.set('from', pathname);
    return NextResponse.redirect(url);
  }

  const response = NextResponse.next();
  
  // Add security headers
  response.headers.set(
    'Strict-Transport-Security',
    'max-age=31536000; includeSubDomains'
  );
  
  return response;
}`,
  },
  {
    id: "next-3",
    title: "Next.js Dynamic Sitemap Generator",
    description: "Generate dynamic sitemaps for SEO optimization",
    technology: "nextjs",
    tags: ["seo", "dynamic-routes"],
    code: `import { GetServerSideProps } from 'next';
import { db } from '@/lib/db';

type Sitemap = {
  url: string;
  lastmod?: string;
  changefreq?: string;
  priority?: number;
};

async function generateSitemap(baseUrl: string): Promise<Sitemap[]> {
  const staticPages = [
    '',
    '/about',
    '/contact',
    '/blog'
  ].map(route => ({
    url: \`\${baseUrl}\${route}\`,
    changefreq: 'weekly',
    priority: route === '' ? 1.0 : 0.8
  }));

  const posts = await db.post.findMany({
    select: {
      slug: true,
      updatedAt: true
    },
    where: {
      published: true
    }
  });

  const dynamicPages = posts.map(post => ({
    url: \`\${baseUrl}/blog/\${post.slug}\`,
    lastmod: post.updatedAt.toISOString(),
    changefreq: 'monthly',
    priority: 0.6
  }));

  return [...staticPages, ...dynamicPages];
}`,
  },
  {
    id: "next-4",
    title: "Next.js Image Gallery with Blur Preview",
    description: "Image gallery with blur-up preview loading",
    technology: "nextjs",
    tags: ["images", "performance"],
    code: `import { useState } from 'react';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';

interface ImageData {
  id: string;
  src: string;
  blurDataUrl: string;
  width: number;
  height: number;
  alt: string;
}

interface GalleryProps {
  images: ImageData[];
}

export default function Gallery({ images }: GalleryProps) {
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {images.map((image) => (
          <motion.div
            key={image.id}
            layoutId={image.id}
            onClick={() => setSelectedImage(image)}
            className="cursor-pointer relative aspect-square"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Image
              src={image.src}
              alt={image.alt}
              fill
              className="object-cover rounded-lg"
              sizes="(max-width: 768px) 50vw,
                     (max-width: 1024px) 33vw,
                     25vw"
              placeholder="blur"
              blurDataURL={image.blurDataUrl}
            />
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedImage(null)}
            className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center"
          >
            <motion.div
              layoutId={selectedImage.id}
              className="relative max-w-4xl max-h-[90vh] w-full mx-4"
            >
              <Image
                src={selectedImage.src}
                alt={selectedImage.alt}
                width={selectedImage.width}
                height={selectedImage.height}
                className="object-contain w-full h-full rounded-lg"
                priority
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}`,
  },
  {
    id: "next-5",
    title: "Next.js Internationalization Setup",
    description: "Complete i18n setup with language switching",
    technology: "nextjs",
    tags: ["i18n", "localization"],
    code: `// next.config.js
const nextConfig = {
  i18n: {
    locales: ['en', 'es', 'fr'],
    defaultLocale: 'en'
  }
};

// lib/i18n/translations.ts
interface Translations {
  [key: string]: {
    [key: string]: string;
  };
}

export const translations: Translations = {
  en: {
    welcome: 'Welcome',
    about: 'About Us',
    contact: 'Contact'
  },
  es: {
    welcome: 'Bienvenidos',
    about: 'Sobre Nosotros',
    contact: 'Contacto'
  },
  fr: {
    welcome: 'Bienvenue',
    about: 'À Propos',
    contact: 'Contact'
  }
};

// lib/i18n/useTranslation.ts
import { useRouter } from 'next/router';

export function useTranslation() {
  const router = useRouter();
  const { locale, locales, defaultLocale } = router;

  const t = (key: string): string => {
    if (!translations[locale!]?.[key]) {
      console.warn(\`Translation key "\${key}" not found for locale "\${locale}"\`);
      return translations[defaultLocale!]?.[key] || key;
    }
    return translations[locale!][key];
  };

  return {
    t,
    locale,
    locales,
    defaultLocale,
    changeLocale: (newLocale: string) => {
      router.push(router.pathname, router.asPath, { locale: newLocale });
    }
  };
}`,
  },

  // TypeScript Snippets
  {
    id: "ts-1",
    title: "Advanced TypeScript Utility Types",
    description: "Collection of useful TypeScript utility types",
    technology: "typescript",
    tags: ["types", "utility"],
    code: `// Deep Partial
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// Deep Required
type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

// Deep Readonly
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// NonNullable nested objects
type DeepNonNullable<T> = {
  [P in keyof T]: T[P] extends object
    ? DeepNonNullable<NonNullable<T[P]>>
    : NonNullable<T[P]>;
};

// Pick nested properties
type NestedPick<T, K extends string> = K extends \`\${infer F}.\${infer R}\`
  ? F extends keyof T
    ? { [P in F]: NestedPick<T[F], R> }
    : never
  : K extends keyof T
  ? { [P in K]: T[K] }
  : never;`,
  },
  {
    id: "ts-2",
    title: "TypeScript State Machine",
    description: "Type-safe state machine implementation",
    technology: "typescript",
    tags: ["patterns", "state-management"],
    code: `type StateConfig<TState extends string, TEvent extends string> = {
  [K in TState]: {
    on?: {
      [E in TEvent]?: TState;
    };
    entry?: () => void;
    exit?: () => void;
  };
};

class StateMachine<TState extends string, TEvent extends string> {
  private currentState: TState;
  private config: StateConfig<TState, TEvent>;

  constructor(initialState: TState, config: StateConfig<TState, TEvent>) {
    this.currentState = initialState;
    this.config = config;
    this.config[initialState].entry?.();
  }

  public getState(): TState {
    return this.currentState;
  }

  public transition(event: TEvent): boolean {
    const currentStateConfig = this.config[this.currentState];
    const nextState = currentStateConfig.on?.[event];

    if (!nextState) {
      return false;
    }

    currentStateConfig.exit?.();
    this.currentState = nextState;
    this.config[nextState].entry?.();

    return true;
  }
}`,
  },
  {
    id: "ts-3",
    title: "TypeScript Decorators",
    description: "Common decorator patterns and implementations",
    technology: "typescript",
    tags: ["decorators", "patterns"],
    code: `// Method Decorator: Logging
function log(
  target: any,
  propertyKey: string,
  descriptor: PropertyDescriptor
) {
  const originalMethod = descriptor.value;

  descriptor.value = function (...args: any[]) {
    console.log(\`Calling \${propertyKey} with args: \${JSON.stringify(args)}\`);
    const result = originalMethod.apply(this, args);
    console.log(\`Method \${propertyKey} returned: \${JSON.stringify(result)}\`);
    return result;
  };

  return descriptor;
}

// Method Decorator: Memoization
function memoize(
  target: any,
  propertyKey: string,
  descriptor: PropertyDescriptor
) {
  const originalMethod = descriptor.value;
  const cache = new Map<string, any>();

  descriptor.value = function (...args: any[]) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = originalMethod.apply(this, args);
    cache.set(key, result);
    return result;
  };

  return descriptor;
}`,
  },
  {
    id: "ts-4",
    title: "TypeScript Event Bus",
    description: "Type-safe event bus implementation",
    technology: "typescript",
    tags: ["events", "patterns"],
    code: `type EventMap = {
  'user:created': { id: string; name: string };
  'user:updated': { id: string; changes: Partial<User> };
  'user:deleted': { id: string };
  'error': { message: string; code: number };
};

type EventKey = keyof EventMap;
type EventCallback<T extends EventKey> = (data: EventMap[T]) => void;

class EventBus {
  private static instance: EventBus;
  private handlers: Map<EventKey, Set<EventCallback<any>>>;

  private constructor() {
    this.handlers = new Map();
  }

  public static getInstance(): EventBus {
    if (!EventBus.instance) {
      EventBus.instance = new EventBus();
    }
    return EventBus.instance;
  }

  public on<T extends EventKey>(
    event: T,
    callback: EventCallback<T>
  ): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }

    this.handlers.get(event)!.add(callback);
    return () => {
      this.handlers.get(event)?.delete(callback);
    };
  }
}`,
  },
  {
    id: "ts-5",
    title: "TypeScript API Client",
    description: "Type-safe API client with interceptors",
    technology: "typescript",
    tags: ["api", "http"],
    code: `type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

interface RequestConfig<T = any> {
  method: HttpMethod;
  url: string;
  data?: T;
  params?: Record<string, string>;
  headers?: Record<string, string>;
}

interface ResponseType<T = any> {
  data: T;
  status: number;
  headers: Record<string, string>;
}

type RequestInterceptor = (
  config: RequestConfig
) => RequestConfig | Promise<RequestConfig>;

type ResponseInterceptor = (
  response: ResponseType
) => ResponseType | Promise<ResponseType>;

class ApiClient {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;
  private requestInterceptors: RequestInterceptor[];
  private responseInterceptors: ResponseInterceptor[];

  constructor(baseURL: string) {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json'
    };
    this.requestInterceptors = [];
    this.responseInterceptors = [];
  }

  public async request<T = any, R = any>(
    config: RequestConfig<T>
  ): Promise<ResponseType<R>> {
    let processedConfig = { ...config };
    
    for (const interceptor of this.requestInterceptors) {
      processedConfig = await interceptor(processedConfig);
    }

    const response = await fetch(
      this.createUrl(processedConfig.url),
      {
        method: processedConfig.method,
        headers: processedConfig.headers,
        body: processedConfig.data ? JSON.stringify(processedConfig.data) : undefined
      }
    );

    return {
      data: await response.json(),
      status: response.status,
      headers: Object.fromEntries(response.headers)
    };
  }
}`,
  },

  // JavaScript Snippets
  {
    id: "js-1",
    title: "Debounce & Throttle",
    description: "Utility functions for rate limiting",
    technology: "javascript",
    tags: ["utils", "performance"],
    code: `function debounce(func, wait) {
  let timeout;
  
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };

    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function throttle(func, wait) {
  let waiting = false;
  let lastArgs = null;
  
  return function executedFunction(...args) {
    if (waiting) {
      lastArgs = args;
      return;
    }
    
    func(...args);
    waiting = true;
    
    setTimeout(() => {
      waiting = false;
      if (lastArgs) {
        executedFunction(...lastArgs);
        lastArgs = null;
      }
    }, wait);
  };
}`,
  },
  {
    id: "js-2",
    title: "Deep Clone",
    description: "Deep clone objects and arrays",
    technology: "javascript",
    tags: ["utils", "objects"],
    code: `function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime());
  }

  if (obj instanceof RegExp) {
    return new RegExp(obj);
  }

  if (obj instanceof Map) {
    const clone = new Map();
    obj.forEach((value, key) => {
      clone.set(deepClone(key), deepClone(value));
    });
    return clone;
  }

  if (obj instanceof Set) {
    const clone = new Set();
    obj.forEach(value => {
      clone.add(deepClone(value));
    });
    return clone;
  }

  const clone = Array.isArray(obj) ? [] : {};

  Object.keys(obj).forEach(key => {
    clone[key] = deepClone(obj[key]);
  });

  return clone;
}`,
  },
  {
    id: "js-3",
    title: "Promise Pool",
    description: "Limit concurrent promises execution",
    technology: "javascript",
    tags: ["async", "performance"],
    code: `async function promisePool(tasks, concurrency) {
  const results = [];
  const executing = new Set();

  async function executeTask(task, index) {
    executing.add(task);
    try {
      const result = await task();
      results[index] = result;
    } catch (error) {
      results[index] = { error };
    }
    executing.delete(task);
  }

  const taskPromises = tasks.map((task, index) => {
    const waitForSlot = async () => {
      while (executing.size >= concurrency) {
        await Promise.race(executing);
      }
      return executeTask(task, index);
    };
    return waitForSlot();
  });

  await Promise.all(taskPromises);
  return results;
}

// Example usage:
const tasks = urls.map(url => async () => {
  const response = await fetch(url);
  return response.json();
});

const results = await promisePool(tasks, 3);`,
  },
  {
    id: "js-4",
    title: "Observable Pattern",
    description: "Implementation of the Observable pattern",
    technology: "javascript",
    tags: ["patterns", "events"],
    code: `class Observable {
  constructor() {
    this.observers = new Set();
  }

  subscribe(observer) {
    this.observers.add(observer);
    return () => this.observers.delete(observer);
  }

  notify(data) {
    this.observers.forEach(observer => observer(data));
  }
}

class Store extends Observable {
  constructor(initialState = {}) {
    super();
    this.state = initialState;
  }

  setState(newState) {
    this.state = { ...this.state, ...newState };
    this.notify(this.state);
  }

  getState() {
    return this.state;
  }
}

// Example usage:
const store = new Store({ count: 0 });

const unsubscribe = store.subscribe(state => {
  console.log('State updated:', state);
});

store.setState({ count: store.getState().count + 1 });`,
  },
  {
    id: "js-5",
    title: "Memoization with Cache",
    description: "Cache function results for better performance",
    technology: "javascript",
    tags: ["performance", "cache"],
    code: `function memoize(fn, options = {}) {
  const {
    maxSize = 1000,
    ttl = 0,
    keyGenerator = (...args) => JSON.stringify(args)
  } = options;

  const cache = new Map();
  const timestamps = new Map();

  function clearExpired() {
    if (!ttl) return;
    
    const now = Date.now();
    for (const [key, timestamp] of timestamps.entries()) {
      if (now - timestamp > ttl) {
        cache.delete(key);
        timestamps.delete(key);
      }
    }
  }

  return function (...args) {
    clearExpired();

    const key = keyGenerator(...args);

    if (cache.has(key)) {
      return cache.get(key);
    }

    const result = fn.apply(this, args);

    if (cache.size continuing from the previous JavaScript memoization snippet...

    if (cache.size >= maxSize) {
      const oldestKey = cache.keys().next().value;
      cache.delete(oldestKey);
      timestamps.delete(oldestKey);
    }

    cache.set(key, result);
    timestamps.set(key, Date.now());

    return result;
  };
}

// Example usage:
const expensiveOperation = memoize(
  (n) => {
    console.log('Computing...');
    return n * 2;
  },
  {
    maxSize: 100,
    ttl: 5000, // 5 seconds
    keyGenerator: (n) => n.toString()
  }
);`,
  },

  // HTML Snippets
  {
    id: "html-1",
    title: "Responsive Navigation Bar",
    description: "Mobile-friendly navigation with hamburger menu",
    technology: "html",
    tags: ["navigation", "responsive"],
    code: `<nav class="bg-white shadow">
  <div class="max-w-7xl mx-auto px-4">
    <div class="flex justify-between h-16">
      <div class="flex">
        <div class="flex-shrink-0 flex items-center">
          <img class="h-8 w-auto" src="/logo.svg" alt="Logo">
        </div>
        <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
          <a href="#" class="border-indigo-500 text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
            Home
          </a>
          <a href="#" class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
            Products
          </a>
          <a href="#" class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
            About
          </a>
          <a href="#" class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
            Contact
          </a>
        </div>
      </div>
      <div class="-mr-2 flex items-center sm:hidden">
        <button type="button" class="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500" aria-controls="mobile-menu" aria-expanded="false">
          <span class="sr-only">Open main menu</span>
          <svg class="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    </div>
  </div>

  <div class="sm:hidden" id="mobile-menu">
    <div class="pt-2 pb-3 space-y-1">
      <a href="#" class="bg-indigo-50 border-indigo-500 text-indigo-700 block pl-3 pr-4 py-2 border-l-4 text-base font-medium">
        Home
      </a>
      <a href="#" class="border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700 block pl-3 pr-4 py-2 border-l-4 text-base font-medium">
        Products
      </a>
      <a href="#" class="border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700 block pl-3 pr-4 py-2 border-l-4 text-base font-medium">
        About
      </a>
      <a href="#" class="border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700 block pl-3 pr-4 py-2 border-l-4 text-base font-medium">
        Contact
      </a>
    </div>
  </div>
</nav>`,
  },
  {
    id: "html-2",
    title: "Hero Section with CTA",
    description: "Responsive hero section with call-to-action",
    technology: "html",
    tags: ["layout", "hero"],
    code: `<div class="relative bg-white overflow-hidden">
  <div class="max-w-7xl mx-auto">
    <div class="relative z-10 pb-8 bg-white sm:pb-16 md:pb-20 lg:max-w-2xl lg:w-full lg:pb-28 xl:pb-32">
      <main class="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
        <div class="sm:text-center lg:text-left">
          <h1 class="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
            <span class="block xl:inline">Data to enrich your</span>
            <span class="block text-indigo-600 xl:inline">online business</span>
          </h1>
          <p class="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
            Anim aute id magna aliqua ad ad non deserunt sunt. Qui irure qui lorem cupidatat commodo. Elit sunt amet fugiat veniam occaecat fugiat aliqua.
          </p>
          <div class="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
            <div class="rounded-md shadow">
              <a href="#" class="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 md:py-4 md:text-lg md:px-10">
                Get started
              </a>
            </div>
            <div class="mt-3 sm:mt-0 sm:ml-3">
              <a href="#" class="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 md:py-4 md:text-lg md:px-10">
                Live demo
              </a>
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>
  <div class="lg:absolute lg:inset-y-0 lg:right-0 lg:w-1/2">
    <img class="h-56 w-full object-cover sm:h-72 md:h-96 lg:w-full lg:h-full" src="https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2850&q=80" alt="">
  </div>
</div>`,
  },
  {
    id: "html-3",
    title: "Contact Form",
    description: "Responsive contact form with validation",
    technology: "html",
    tags: ["forms", "validation"],
    code: `<div class="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
  <div class="relative py-3 sm:max-w-xl sm:mx-auto">
    <div class="relative px-4 py-10 bg-white mx-8 md:mx-0 shadow rounded-3xl sm:p-10">
      <div class="max-w-md mx-auto">
        <div class="divide-y divide-gray-200">
          <div class="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
            <div class="flex flex-col">
              <label class="leading-loose">Name</label>
              <input type="text" class="px-4 py-2 border focus:ring-gray-500 focus:border-gray-900 w-full sm:text-sm border-gray-300 rounded-md focus:outline-none text-gray-600" placeholder="Your name">
            </div>
            <div class="flex flex-col">
              <label class="leading-loose">Email</label>
              <input type="email" class="px-4 py-2 border focus:ring-gray-500 focus:border-gray-900 w-full sm:text-sm border-gray-300 rounded-md focus:outline-none text-gray-600" placeholder="your@email.com">
            </div>
            <div class="flex flex-col">
              <label class="leading-loose">Message</label>
              <textarea class="px-4 py-2 border focus:ring-gray-500 focus:border-gray-900 w-full sm:text-sm border-gray-300 rounded-md focus:outline-none text-gray-600" rows="4" placeholder="Your message"></textarea>
            </div>
          </div>
          <div class="pt-4 flex items-center space-x-4">
            <button class="flex justify-center items-center w-full text-white px-4 py-3 rounded-md focus:outline-none bg-blue-500 hover:bg-blue-600">
              Send Message
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>`,
  },
  {
    id: "html-4",
    title: "Pricing Table",
    description: "Responsive pricing table with multiple tiers",
    technology: "html",
    tags: ["pricing", "cards"],
    code: `<div class="bg-white py-12">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="lg:text-center">
      <h2 class="text-base text-indigo-600 font-semibold tracking-wide uppercase">Pricing</h2>
      <p class="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
        Choose the right plan for you
      </p>
    </div>

    <div class="mt-10">
      <div class="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
        <!-- Basic Plan -->
        <div class="border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
          <div class="p-6">
            <h2 class="text-lg leading-6 font-medium text-gray-900">Basic</h2>
            <p class="mt-4 text-sm text-gray-500">All the basics for starting a new business</p>
            <p class="mt-8">
              <span class="text-4xl font-extrabold text-gray-900">$29</span>
              <span class="text-base font-medium text-gray-500">/mo</span>
            </p>
            <a href="#" class="mt-8 block w-full bg-indigo-600 border border-transparent rounded-md py-2 text-sm font-semibold text-white text-center hover:bg-indigo-700">
              Buy Basic
            </a>
          </div>
          <div class="pt-6 pb-8 px-6">
            <h3 class="text-xs font-medium text-gray-900 tracking-wide uppercase">What's included</h3>
            <ul role="list" class="mt-6 space-y-4">
              <li class="flex space-x-3">
                <svg class="flex-shrink-0 h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm text-gray-500">5 Products</span>
              </li>
              <li class="flex space-x-3">
                <svg class="flex-shrink-0 h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm text-gray-500">Basic Analytics</span>
              </li>
              <li class="flex space-x-3">
                <svg class="flex-shrink-0 h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm text-gray-500">48-hour support response time</span>
              </li>
            </ul>
          </div>
        </div>

        <!-- Pro Plan -->
        <div class="border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
          <div class="p-6">
            <h2 class="text-lg leading-6 font-medium text-gray-900">Pro</h2>
            <p class="mt-4 text-sm text-gray-500">Professional grade features for growing businesses</p>
            <p class="mt-8">
              <span class="text-4xl font-extrabold text-gray-900">$99</span>
              <span class="text-base font-medium text-gray-500">/mo</span>
            </p>
            <a href="#" class="mt-8 block w-full bg-indigo-600 border border-transparent rounded-md py-2 text-sm font-semibold text-white text-center hover:bg-indigo-700">
              Buy Pro
            </a>
          </div>
          <div class="pt-6 pb-8 px-6">
            <h3 class="text-xs font-medium text-gray-900 tracking-wide uppercase">What's included</h3>
            <ul role="list" class="mt-6 space-y-4">
              <li class="flex space-x-3">
                <svg class="flex-shrink-0 h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm text-gray-500">Unlimited Products</span>
              </li>
              <li class="flex space-x-3">
                <svg class="flex-shrink-0 h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm text-gray-500">Advanced Analytics</span>
              </li>
              <li class="flex space-x-3">
                <svg class="flex-shrink-0 h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm text-gray-500">24/7 support</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>`,
  },
  {
    id: "html-5",
    title: "Footer with Newsletter",
    description: "Responsive footer with newsletter subscription",
    technology: "html",
    tags: ["footer", "newsletter"],
    code: `<footer class="bg-gray-800">
  <div class="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8">
    <div class="xl:grid xl:grid-cols-3 xl:gap-8">
      <div class="grid grid-cols-2 gap-8 xl:col-span-2">
        <div class="md:grid md:grid-cols-2 md:gap-8">
          <div>
            <h3 class="text-sm font-semibold text-gray-400 tracking-wider uppercase">
              Solutions
            </h3>
            <ul role="list" class="mt-4 space-y-4">
              <li>
                <a href="#" class="text-base text-gray-300 hover:text-white">
                  Marketing
                </a>
              </li>
              <li>
                <a href="#" class="text-base text-gray-300 hover:text-white">
                  Analytics
                </a>
              </li>
              <li>
                <a href="#" class="text-base text-gray-300 hover:text-white">
                  Commerce
                </a>
              </li>
            </ul>
          </div>
          <div class="mt-12 md:mt-0">
            <h3 class="text-sm font-semibold text-gray-400 tracking-wider uppercase">
              Support
            </h3>
            <ul role="list" class="mt-4 space-y-4">
              <li>
                <a href="#" class="text-base text-gray-300 hover:text-white">
                  Pricing
                </a>
              </li>
              <li>
                <a href="#" class="text-base text-gray-300 hover:text-white">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" class="text-base text-gray-300 hover:text-white">
                  Guides
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div class="mt-8 xl:mt-0">
        <h3 class="text-sm font-semibold text-gray-400 tracking-wider uppercase">
          Subscribe to our newsletter
        </h3>
        <p class="mt-4 text-base text-gray-300">
          The latest news, articles, and resources, sent to your inbox weekly.
        </p>
        <form class="mt-4 sm:flex sm:max-w-md">
          <label for="email-address" class="sr-only">Email address</label>
          <input type="email" name="email-address" id="email-address" autocomplete="email" required class="appearance-none min-w-0 w-full bg-white border border-transparent rounded-md py-2 px-4 text-base text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-white focus:border-white focus:placeholder-gray-400" placeholder="Enter your email">
          <div class="mt-3 rounded-md sm:mt-0 sm:ml-3 sm:flex-shrink-0">
            <button type="submit" class="w-full bg-indigo-500 border border-transparent rounded-md py-2 px-4 flex items-center justify-center text-base font-medium text-white hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500">
              Subscribe
            </button>
          </div>
        </form>
      </div>
    </div>
    <div class="mt-8 border-t border-gray-700 pt-8 md:flex md:items-center md:justify-between">
      <div class="flex space-x-6 md:order-2">
        <a href="#" class="text-gray-400 hover:text-gray-300">
          <span class="sr-only">Facebook</span>
          <svg class="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path fill-rule="evenodd" d="M22 12c0-5.523-4.477-10-10-10S2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.988C18.343 21.128 22 16.991 22 12z" clip-rule="evenodd" />
          </svg>
        </a>
        <a href="#" class="text-gray-400 hover:text-gray-300">
          <span class="sr-only">Twitter</span>
          <svg class="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
          </svg>
        </a>
      </div>
      <p class="mt-8 text-base text-gray-400 md:mt-0 md:order-1">
        &copy; 2023 Your Company, Inc. All rights reserved.
      </p>
    </div>
  </div>
</footer>`,
  },
];

function App() {
  const [selectedTech, setSelectedTech] = useState<Technology>("react");
  const [searchQuery, setSearchQuery] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copyToClipboard = async (code: string, id: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const filteredSnippets = snippets.filter((snippet) => {
    const matchesTech = snippet.technology === selectedTech;
    const matchesSearch =
      searchQuery.toLowerCase().trim() === "" ||
      snippet.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      snippet.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      snippet.tags?.some((tag) =>
        tag.toLowerCase().includes(searchQuery.toLowerCase())
      );

    return matchesTech && matchesSearch;
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Code2 className="w-8 h-8 text-blue-600" />
              <h1 className="text-xl font-bold text-gray-900">
                Code Snippet Generator
              </h1>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search snippets..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Technology Selector */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Filter className="w-5 h-5 text-gray-500" />
            <h2 className="text-lg font-semibold text-gray-900">
              Select Technology
            </h2>
          </div>
          <div className="flex flex-wrap gap-4">
            {(
              [
                "react",
                "nextjs",
                "html",
                "typescript",
                "javascript",
              ] as Technology[]
            ).map((tech) => (
              <button
                key={tech}
                onClick={() => setSelectedTech(tech)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedTech === tech
                    ? "bg-blue-600 text-white shadow-sm"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}>
                {tech === "nextjs"
                  ? "Next.js"
                  : tech.charAt(0).toUpperCase() + tech.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Snippets Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {filteredSnippets.map((snippet) => (
            <div
              key={snippet.id}
              className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900">
                  {snippet.title}
                </h3>
                <p className="mt-2 text-gray-600">{snippet.description}</p>
                {snippet.tags && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {snippet.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 text-xs font-medium text-blue-600 bg-blue-50 rounded-full">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="relative">
                <pre className="bg-gray-900 p-4 overflow-x-auto">
                  <code className="text-gray-100 text-sm">{snippet.code}</code>
                </pre>
                <button
                  onClick={() => copyToClipboard(snippet.code, snippet.id)}
                  className="absolute top-2 right-2 p-2 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors"
                  title="Copy code">
                  {copiedId === snippet.id ? (
                    <CheckCircle2 className="w-5 h-5 text-green-400" />
                  ) : (
                    <Copy className="w-5 h-5" />
                  )}
                </button>{" "}
              </div>
            </div>
          ))}
        </div>

        {filteredSnippets.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">
              No snippets found. Try adjusting your search.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
