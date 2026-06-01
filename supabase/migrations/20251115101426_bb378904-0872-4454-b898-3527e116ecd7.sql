-- Create profiles table
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT,
  email TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view their own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile"
  ON public.profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

-- Create records table for upload tracking
CREATE TABLE public.records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  disease_type TEXT NOT NULL,
  file_type TEXT NOT NULL,
  file_url TEXT,
  upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  status TEXT DEFAULT 'pending'
);

-- Enable RLS on records
ALTER TABLE public.records ENABLE ROW LEVEL SECURITY;

-- Records policies
CREATE POLICY "Users can view their own records"
  ON public.records FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own records"
  ON public.records FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own records"
  ON public.records FOR UPDATE
  USING (auth.uid() = user_id);

-- Create predictions table
CREATE TABLE public.predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  record_id UUID REFERENCES public.records(id) ON DELETE CASCADE,
  disease_type TEXT NOT NULL,
  prediction_result JSONB,
  confidence_score FLOAT,
  explainability_data JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on predictions
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- Predictions policies
CREATE POLICY "Users can view their own predictions"
  ON public.predictions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own predictions"
  ON public.predictions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Create chatlogs table
CREATE TABLE public.chatlogs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  message TEXT NOT NULL,
  response TEXT,
  intent TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on chatlogs
ALTER TABLE public.chatlogs ENABLE ROW LEVEL SECURITY;

-- Chatlogs policies
CREATE POLICY "Users can view their own chatlogs"
  ON public.chatlogs FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert chatlogs"
  ON public.chatlogs FOR INSERT
  WITH CHECK (true);

-- Create trigger for profiles
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'full_name', '')
  );
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- Add trigger to profiles
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();